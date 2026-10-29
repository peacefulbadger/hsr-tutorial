"""Resolve generic Actions into concrete coordinates.

The decision layer speaks in ids ("continue", "basic_atk"), never in pixels. This
is where an id becomes a point, and where a stale id becomes an explicit failure
rather than a tap into empty space.
"""

from __future__ import annotations

import numpy as np

from adapters.game_hsr.features import extract
from adapters.game_hsr.perception import HsrPerception
from adapters.game_hsr.render import render_state_for_mica
from core.action import Action
from core.state import GameState


class HsrAdapter:
    """Concrete GameAdapter for Honkai Star Rail."""

    def __init__(self, perception: HsrPerception | None = None) -> None:
        self.perception = perception or HsrPerception()

    # ---------------------------------------------------------------- perception

    def perceive(self, frame: np.ndarray) -> GameState:
        return self.perception.perceive(frame)

    # -------------------------------------------------------------------- render

    def render_state_for_mica(self, state: GameState) -> str:
        return render_state_for_mica(state)

    def candidate_actions(self, state: GameState) -> dict[str, str]:
        """Ordered id -> description. Order is not a hint; the caller shuffles."""
        candidates: dict[str, str] = {}

        if state.screen_type == "combat":
            candidates["basic_atk"] = "Use the basic attack on the current target"
            candidates["blast"] = "Use the blast action (hits adjacent enemies)"
            candidates["wait"] = "Do nothing and wait"
        elif state.screen_type == "dialogue":
            candidates["continue"] = "Advance the dialogue"
            candidates["wait"] = "Do nothing and wait"
        elif state.screen_type == "menu":
            for element in state.interactive_elements:
                candidates[element.id] = f"Press {element.label}"
            candidates["wait"] = "Do nothing and wait"
        else:
            for element in state.interactive_elements:
                candidates[element.id] = f"Press {element.label}"
            candidates["wait"] = "Do nothing and wait"

        return candidates

    # ------------------------------------------------------------------- resolve

    def resolve(self, action: Action, state: GameState) -> Action:
        if action.type != "tap":
            return action
        if action.bbox is not None:
            return action
        if action.target_id is None:
            raise ValueError("tap action has neither bbox nor target_id")
        if action.target_id == "wait":
            return Action(type="wait", duration_ms=action.duration_ms)
        element = state.element(action.target_id)
        if element is None:
            raise ValueError(
                f"target {action.target_id!r} is not present on screen "
                f"{state.screen_type!r}"
            )
        return action.model_copy(update={"bbox": list(element.bbox)})


__all__ = ["HsrAdapter", "extract"]
