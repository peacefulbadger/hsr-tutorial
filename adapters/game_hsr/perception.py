"""HSR perception: frame -> generic GameState.

Screen-type decision order (cheap first, OCR only when it is needed):

  1. near-black frame                 -> loading
  2. teal action ring bottom-right    -> combat        (one pass over pixels)
  3. OCR the frame, then read text:
       banner says ATK/skill/target   -> combat
       title-screen keywords          -> menu
       subtitle line in the bottom band -> dialogue
  4. dark and detail-free             -> transition
  5. otherwise                        -> unknown

The first version classified on region brightness alone and scored 33%. The
reason is worth keeping: the *scene* behind the HUD dominates pixel statistics.
The dialogue band is 93% dark pixels on one frame and 71% on another depending on
what is behind it, and combat loses its teal ring entirely during attack
animations. Text is the stable signal; HUD colour is only a fast lane.

Engine note: the plan specifies PaddleOCR. PaddlePaddle would not install here --
it downgrades numpy, which breaks the pinned opencv build, then dies on a locked
cv2.pyd. RapidOCR runs the same PP-OCR models on onnxruntime and installs
cleanly. `OcrEngine` is the seam; swapping in paddleocr means one class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from adapters.game_hsr import regions as R
from adapters.game_hsr.features import Features, extract
from core.state import GameState, InteractiveElement

SCREEN_TYPES = (
    "loading",
    "dialogue",
    "dialogue_choice",
    "combat",
    "reward",
    "menu",
    "transition",
    "unknown",
)

MENU_KEYWORDS = ("startgame", "taptostart", "logout", "notices", "settings")
COMBAT_KEYWORDS = (
    "basicatk", "selecttheenemy", "switchtoskills", "singletarget", "blast", "attack",
)


def _norm(text: str) -> str:
    """Lowercase, drop punctuation. OCR renders 'Start Game' as 'Start.Game'
    and 'Use Basic ATK on' as 'Use BasicATKon', so exact substrings are unsafe."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


@dataclass
class TextBox:
    text: str
    confidence: float
    bbox: list[int]

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


class OcrEngine(Protocol):
    def read(self, image_bgr: np.ndarray) -> list[TextBox]: ...


class RapidOcr:
    """RapidOCR (PP-OCR models, onnxruntime)."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        self._min_confidence = min_confidence

    def read(self, image_bgr: np.ndarray) -> list[TextBox]:
        result, _elapsed = self._engine(image_bgr)
        boxes: list[TextBox] = []
        for item in result or []:
            points, text, score = item[0], item[1], float(item[2])
            if score < self._min_confidence or not str(text).strip():
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            boxes.append(TextBox(text=str(text).strip(), confidence=score, bbox=[x, y, w, h]))
        return boxes


def _inside(bbox: list[int], region: tuple[int, int, int, int]) -> bool:
    x, y, w, h = bbox
    rx, ry, rw, rh = region
    cx, cy = x + w / 2, y + h / 2
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


class HsrPerception:
    """Frame in, GameState out. Stateless between frames."""

    def __init__(self, ocr: OcrEngine | None = None, *, ocr_enabled: bool = True) -> None:
        self._ocr = ocr
        self._ocr_enabled = ocr_enabled

    # ------------------------------------------------------------- classification

    def analyze(
        self, frame: np.ndarray, features: Features | None = None
    ) -> tuple[str, float, Features, list[TextBox]]:
        feats = features or extract(frame, R.FEATURE_REGIONS)

        if feats.mean_brightness < 12.0:
            return "loading", 0.95, feats, []

        boxes: list[TextBox] = []
        if self._ocr is not None and self._ocr_enabled:
            boxes = self._ocr.read(frame)

        # Fast lane: the teal action ring is unambiguous when present.
        if feats.teal_ring_pixels > 0.04:
            return "combat", 0.92, feats, boxes

        # Combat prompts sit above the subtitle strip (the banner moves between
        # tutorial steps), so search everything above it.
        upper_text = _norm(" ".join(b.text for b in boxes if b.bbox[1] < R.DIALOGUE_BAND[1] - 20))
        if upper_text and any(keyword in upper_text for keyword in COMBAT_KEYWORDS):
            return "combat", 0.85, feats, boxes

        joined = _norm(" ".join(b.text for b in boxes))
        if any(keyword in joined for keyword in MENU_KEYWORDS):
            return "menu", 0.8, feats, boxes

        band = [b for b in boxes if b.bbox[1] >= R.DIALOGUE_BAND[1] - 20]
        band_text = " ".join(b.text for b in band).strip()
        if len(band_text) >= 6:
            return "dialogue", 0.85, feats, boxes

        if feats.edge_density < 0.02:
            return "transition", 0.55, feats, boxes
        return "unknown", 0.3, feats, boxes

    def classify(self, frame: np.ndarray) -> tuple[str, float, Features]:
        screen_type, confidence, feats, _boxes = self.analyze(frame)
        return screen_type, confidence, feats

    # ------------------------------------------------------------------ perception

    def perceive(
        self,
        frame: np.ndarray,
        *,
        frame_path: str = "",
        timestamp_ms: int = 0,
        frame_age_ms: int = 0,
    ) -> GameState:
        screen_type, confidence, _feats, boxes = self.analyze(frame)

        elements: list[InteractiveElement] = []
        choices: list[str] = []
        numeric: dict[str, float] = {}

        if screen_type == "dialogue":
            band_boxes = [b for b in boxes if b.bbox[1] >= R.DIALOGUE_BAND[1] - 20]
            speaker, body = self._split_speaker(band_boxes)
            x, y, w, h = R.CONTINUE_CHEVRON
            elements.append(InteractiveElement(
                id="continue", label="continue", bbox=[x, y, w, h], kind="button", confidence=0.9,
            ))
            if speaker:
                numeric["speaker_known"] = 1.0
            text_blocks = [body] if body else []

        elif screen_type == "combat":
            text_blocks = [b.text for b in boxes if _inside(b.bbox, R.TUTORIAL_BANNER)]
            x, y, w, h = R.ACTION_BUTTON
            elements.append(InteractiveElement(
                id="basic_atk", label="Basic ATK", bbox=[x, y, w, h], kind="button", confidence=0.9,
            ))
            bx, by, bw, bh = R.BLAST_BUTTON
            elements.append(InteractiveElement(
                id="blast", label="Blast", bbox=[bx, by, bw, bh], kind="button", confidence=0.7,
            ))
            if text_blocks:
                numeric["tutorial_prompt"] = 1.0

        elif screen_type == "menu":
            text_blocks = [b.text for b in boxes]
            if any("startgame" in _norm(t) for t in text_blocks):
                elements.append(InteractiveElement(
                    id="start_game", label="Start Game", bbox=[560, 495, 160, 45],
                    kind="button", confidence=0.85,
                ))

        else:
            text_blocks = []

        return GameState(
            timestamp_ms=timestamp_ms,
            frame_path=frame_path,
            screen_type=screen_type,
            text_blocks=[t for t in text_blocks if t],
            choices=choices,
            interactive_elements=elements,
            numeric_indicators=numeric,
            frame_age_ms=frame_age_ms,
            perception_confidence=confidence,
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _split_speaker(boxes: list[TextBox]) -> tuple[str | None, str | None]:
        """The band holds a small speaker plate above the spoken line."""
        if not boxes:
            return None, None
        ordered = sorted(boxes, key=lambda b: b.bbox[1])
        if len(ordered) == 1:
            return None, ordered[0].text
        speaker = min(ordered, key=lambda b: len(b.text))
        if len(speaker.text) >= 30:
            return None, " ".join(b.text for b in ordered)
        body = " ".join(b.text for b in ordered if b is not speaker)
        return speaker.text, body
