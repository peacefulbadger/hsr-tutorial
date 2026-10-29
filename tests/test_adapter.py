"""Adapter contract tests: render, candidate extraction, id -> coordinate resolution."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from adapters.game_hsr.perception import HsrPerception
from adapters.game_hsr.resolve import HsrAdapter
from core.action import Action
from core.state import GameState, InteractiveElement

FIXTURES = Path(__file__).parent / "fixtures"


def ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


needs_ocr = pytest.mark.skipif(not ocr_available(), reason="rapidocr not installed")


@pytest.fixture(scope="module")
def adapter() -> HsrAdapter:
    from adapters.game_hsr.perception import RapidOcr

    return HsrAdapter(HsrPerception(ocr=RapidOcr()))


def state_for(screen_type: str = "dialogue") -> GameState:
    return GameState(
        timestamp_ms=0,
        frame_path="x.jpg",
        screen_type=screen_type,
        text_blocks=["Some line of dialogue"],
        interactive_elements=[
            InteractiveElement(id="continue", label="continue", bbox=[620, 655, 45, 50],
                               kind="button", confidence=0.9),
        ],
        perception_confidence=0.9,
    )


# ------------------------------------------------------------------- resolution


def test_resolve_turns_id_into_centre_coordinates():
    state = state_for()
    resolved = HsrAdapter().resolve(Action(type="tap", target_id="continue"), state)
    assert resolved.bbox == [620, 655, 45, 50]
    assert resolved.point() == (642, 680)


def test_resolve_rejects_absent_target():
    state = state_for()
    with pytest.raises(ValueError, match="not present"):
        HsrAdapter().resolve(Action(type="tap", target_id="basic_atk"), state)


def test_resolve_passes_through_non_tap_actions():
    action = Action(type="wait", duration_ms=250)
    assert HsrAdapter().resolve(action, state_for()) is action


def test_resolve_maps_wait_target_to_wait_action():
    resolved = HsrAdapter().resolve(Action(type="tap", target_id="wait"), state_for())
    assert resolved.type == "wait"


def test_resolve_is_idempotent_on_resolved_actions():
    state = state_for()
    once = HsrAdapter().resolve(Action(type="tap", target_id="continue"), state)
    twice = HsrAdapter().resolve(once, state)
    assert once.bbox == twice.bbox


# --------------------------------------------------------------------- rendering


def test_render_is_compact_and_readable():
    text = HsrAdapter().render_state_for_mica(state_for())
    assert "screen: dialogue" in text
    assert "continue" in text
    assert len(text) < 400, "state rendering must stay small"


def test_render_drops_geometry():
    text = HsrAdapter().render_state_for_mica(state_for())
    assert "620" not in text, "pixel boxes are actuator detail, not model input"


def test_render_includes_choices_when_present():
    state = state_for("dialogue_choice")
    state.choices = ["Yes", "No"]
    text = HsrAdapter().render_state_for_mica(state)
    assert "Yes" in text and "No" in text


# -------------------------------------------------------------------- candidates


def test_candidates_for_dialogue():
    candidates = HsrAdapter().candidate_actions(state_for())
    assert set(candidates) == {"continue", "wait"}


def test_candidates_for_combat():
    state = state_for("combat")
    state.interactive_elements = [
        InteractiveElement(id="basic_atk", label="Basic ATK", bbox=[950, 480, 180, 180]),
        InteractiveElement(id="blast", label="Blast", bbox=[1120, 480, 120, 110]),
    ]
    candidates = HsrAdapter().candidate_actions(state)
    assert {"basic_atk", "blast", "wait"} == set(candidates)
    assert all(isinstance(v, str) and v for v in candidates.values())


# ------------------------------------------------------------- end to end wiring


@needs_ocr
def test_perceive_then_candidates_then_resolve(adapter):
    frame = cv2.imread(str(FIXTURES / "combat.jpg"))
    state = adapter.perceive(frame)
    candidates = adapter.candidate_actions(state)
    assert "basic_atk" in candidates

    resolved = adapter.resolve(Action(type="tap", target_id="basic_atk"), state)
    x, y = resolved.point()
    assert 0 <= x < 1280 and 0 <= y < 720
