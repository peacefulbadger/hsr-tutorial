"""Perception tests over saved frames. No emulator, no adb, no network.

Fixtures are four committed frames, one per known screen type. The OCR engine is
optional: if rapidocr is not installed the OCR-dependent tests skip rather than
fail, so the suite still runs on a bare checkout.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from adapters.game_hsr import regions as R
from adapters.game_hsr.features import extract
from adapters.game_hsr.perception import HsrPerception, SCREEN_TYPES
from core.action import Action
from core.state import GameState, InteractiveElement

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    image = cv2.imread(str(FIXTURES / f"{name}.jpg"))
    assert image is not None, f"missing fixture {name}"
    return image


def ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


needs_ocr = pytest.mark.skipif(not ocr_available(), reason="rapidocr not installed")


@pytest.fixture(scope="module")
def perception() -> HsrPerception:
    from adapters.game_hsr.perception import RapidOcr

    return HsrPerception(ocr=RapidOcr())


# ------------------------------------------------------------------- contracts


def test_regions_fit_the_frame():
    for name in ("DIALOGUE_BAND", "DIALOGUE_TEXT", "ACTION_BUTTON", "PARTY_PANEL", "ENEMY_HP"):
        x, y, w, h = getattr(R, name)
        assert x >= 0 and y >= 0
        assert x + w <= R.FRAME_WIDTH, f"{name} overflows width"
        assert y + h <= R.FRAME_HEIGHT, f"{name} overflows height"


def test_game_state_rejects_bad_element_kind():
    with pytest.raises(Exception):
        InteractiveElement(id="x", label="y", bbox=[0, 0, 1, 1], kind="banana")


def test_action_point_is_bbox_centre():
    action = Action(type="tap", bbox=[100, 200, 40, 20])
    assert action.point() == (120, 210)


def test_tap_action_requires_a_target():
    with pytest.raises(Exception):
        Action(type="tap")


# -------------------------------------------------------------------- features


def test_loading_frame_is_dark():
    feats = extract(load("loading"), R.FEATURE_REGIONS)
    assert feats.mean_brightness < 12.0


def test_combat_frame_has_teal_ring():
    feats = extract(load("combat"), R.FEATURE_REGIONS)
    assert feats.teal_ring_pixels > 0.04


def test_dialogue_frame_has_no_teal_ring():
    feats = extract(load("dialogue"), R.FEATURE_REGIONS)
    assert feats.teal_ring_pixels < 0.01


# ------------------------------------------------------------------ classifier


@needs_ocr
@pytest.mark.parametrize("name,expected", [
    ("dialogue", "dialogue"),
    ("combat", "combat"),
    ("menu", "menu"),
    ("loading", "loading"),
    ("reward", "reward"),
])
def test_screen_type_on_fixtures(perception, name, expected):
    screen_type, confidence, _features = perception.classify(load(name))
    assert screen_type == expected, f"{name} classified as {screen_type}"
    assert confidence >= 0.5


def test_loading_needs_no_ocr():
    """A black frame is decided before OCR runs, so it works with no engine."""
    perception = HsrPerception(ocr=None, ocr_enabled=False)
    screen_type, confidence, _ = perception.classify(load("loading"))
    assert screen_type == "loading"
    assert confidence > 0.9


@needs_ocr
def test_dialogue_perception_yields_a_continue_element(perception):
    state = perception.perceive(load("dialogue"), frame_path="x.jpg")
    assert isinstance(state, GameState)
    assert state.screen_type == "dialogue"
    ids = [element.id for element in state.interactive_elements]
    assert "continue" in ids
    assert state.text_blocks, "dialogue should surface the spoken line"
    assert len(" ".join(state.text_blocks)) > 10


@needs_ocr
def test_combat_perception_exposes_actions(perception):
    state = perception.perceive(load("combat"), frame_path="x.jpg")
    assert state.screen_type == "combat"
    ids = [element.id for element in state.interactive_elements]
    assert "basic_atk" in ids
    assert state.numeric_indicators.get("tutorial_prompt") == 1.0


@needs_ocr
def test_menu_perception_exposes_start_game(perception):
    state = perception.perceive(load("menu"), frame_path="x.jpg")
    assert state.screen_type == "menu"
    labels = [element.label for element in state.interactive_elements]
    assert any("start" in label.lower() for label in labels)


def test_perceive_is_stateless(perception):
    first = perception.perceive(load("combat"), frame_path="a.jpg")
    perception.perceive(load("dialogue"), frame_path="b.jpg")
    again = perception.perceive(load("combat"), frame_path="a.jpg")
    assert first.screen_type == again.screen_type
    assert [e.id for e in first.interactive_elements] == [e.id for e in again.interactive_elements]


def test_screen_types_taxonomy_is_the_documented_one():
    assert "dialogue_choice" in SCREEN_TYPES
    assert "reward" in SCREEN_TYPES
    assert len(SCREEN_TYPES) == 8


@needs_ocr
def test_black_frame_is_ambiguous_between_loading_and_transition(perception):
    """A single black frame cannot tell a scene transition from a load.

    `loading` and `transition` are the same observation; separating them needs
    temporal context (how long has it been dark), which belongs to the
    orchestrator, not to a stateless frame classifier. Recording the ambiguity
    here so the taxonomy mismatch is a known quantity rather than a surprise.
    """
    screen_type, _confidence, _features = perception.classify(load("transition"))
    assert screen_type in {"loading", "transition"}


@needs_ocr
def test_reward_outranks_the_combat_fast_lane(perception):
    """The result screen still has the combat HUD behind it, so the reward rule
    must run before the teal-ring check."""
    state = perception.perceive(load("reward"), frame_path="x.jpg")
    assert state.screen_type == "reward"
    assert any("Battle Over" in text for text in state.text_blocks)

@needs_ocr
def test_exploration_is_not_mistaken_for_dialogue(perception):
    """Regression: the UID string at the bottom-left sits at subtitle height but
    outside the band's width. Matching on y alone made every exploration frame
    classify as `dialogue`, which is most of the game's playtime.

    `unknown` is the honest answer here: the plan's taxonomy has no
    `exploration` class, which is itself a gap -- walking around is the state the
    agent spends most of its time in.
    """
    screen_type, _confidence, _features = perception.classify(load("exploration"))
    assert screen_type != "dialogue"
    assert screen_type in {"unknown", "transition"}