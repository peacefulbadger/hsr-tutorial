"""Game-agnostic state schema.

This is the only vocabulary `core/` knows about. It is deliberately shaped like a
generic UI snapshot -- text, choices, interactive elements, numeric indicators --
so that another game's adapter can produce it without pretending to be HSR.

Frozen contract (plan section 5). Adapters may add fields in their own subclass,
but `core/` sees exactly these.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ElementKind = Literal["button", "choice", "target", "other"]


class InteractiveElement(BaseModel):
    id: str
    label: str
    bbox: list[int] = Field(description="[x, y, w, h] in 1280x720 framebuffer space")
    kind: ElementKind = "other"
    confidence: float = 1.0

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


class GameState(BaseModel):
    timestamp_ms: int
    frame_path: str
    screen_type: str = Field(description="adapter-supplied taxonomy")
    text_blocks: list[str] = Field(default_factory=list)
    choices: list[str] = Field(default_factory=list)
    interactive_elements: list[InteractiveElement] = Field(default_factory=list)
    numeric_indicators: dict[str, float] = Field(default_factory=dict)
    frame_age_ms: int = 0
    perception_confidence: float = 0.0

    def element(self, element_id: str) -> InteractiveElement | None:
        return next((e for e in self.interactive_elements if e.id == element_id), None)

    def elements_of_kind(self, kind: ElementKind) -> list[InteractiveElement]:
        return [e for e in self.interactive_elements if e.kind == kind]
