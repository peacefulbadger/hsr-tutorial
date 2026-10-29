"""The only per-game surface.

`core/` depends on this Protocol and nothing else. An adapter is injected, so
adding a game means writing one of these and touching no core module.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from core.action import Action
from core.state import GameState


class GameAdapter(Protocol):
    def perceive(self, frame: np.ndarray) -> GameState: ...

    def render_state_for_mica(self, state: GameState) -> str: ...

    def candidate_actions(self, state: GameState) -> dict[str, str]:
        """id -> human-readable description of what that action does."""
        ...

    def resolve(self, action: Action, state: GameState) -> Action:
        """Turn ids into concrete coordinates. Returns an Action the actuator can
        execute unchanged."""
        ...
