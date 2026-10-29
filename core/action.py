"""Game-agnostic action vocabulary.

`tap | swipe | wait | back | custom` is the whole alphabet. No game-specific verb
appears here; an adapter resolves ids and bboxes into coordinates before the
actuator sees them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ActionType = Literal["tap", "swipe", "wait", "back", "custom"]


class Action(BaseModel):
    type: ActionType
    target_id: str | None = None
    bbox: list[int] | None = Field(default=None, description="[x, y, w, h]")
    direction: Literal["up", "down", "left", "right"] | None = None
    duration_ms: int = 100

    @model_validator(mode="after")
    def _needs_a_target(self) -> "Action":
        if self.type == "tap" and self.target_id is None and self.bbox is None:
            raise ValueError("tap needs target_id or bbox")
        return self

    def point(self) -> tuple[int, int] | None:
        """Centre of the bbox, in framebuffer pixels."""
        if self.bbox is None:
            return None
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2
