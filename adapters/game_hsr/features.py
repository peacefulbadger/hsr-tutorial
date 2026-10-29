"""Cheap visual features for screen-type classification.

Everything here is a single pass over a 1280x720 BGR frame: no OCR, no templates,
no model. OCR is expensive (~100-300 ms/frame) and only needed for labels, so the
classifier leans on colour and layout, and OCR is used to fill in text.

Regions are HSR-specific and live in `regions.py`; this module takes whatever
regions it is given.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass(frozen=True)
class Features:
    mean_brightness: float
    # Dialogue: a translucent dark band behind the subtitle, with bright text.
    dialogue_band_dark: float
    dialogue_text_pixels: float
    # Combat: the teal action-button ring bottom-right, and the party HP panel left.
    teal_ring_pixels: float
    party_panel_bright: float
    # Combat HUD: enemy HP bars are thin bright horizontal bars over mid-screen.
    enemy_hp_bar_pixels: float
    # Generic motion/text density, useful for transition detection.
    edge_density: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _count(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask))


def extract(frame_bgr: np.ndarray, regions: dict[str, tuple[int, int, int, int]]) -> Features:
    """Compute features for one frame. `regions` maps name -> (x, y, w, h)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]

    def crop(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x, y, w, h = regions[name]
        return (
            frame_bgr[y:y + h, x:x + w],
            value[y:y + h, x:x + w],
            saturation[y:y + h, x:x + w],
            hue[y:y + h, x:x + w],
        )

    _, band_v, band_s, _ = crop("dialogue_band")
    band_area = float(band_v.size)
    dark_band = _count(band_v < 75) / band_area

    _, text_v, text_s, _ = crop("dialogue_text")
    text_pixels = _count((text_v > 165) & (text_s < 70)) / float(text_v.size)

    _, ring_v, ring_s, ring_h = crop("action_button")
    # Teal/cyan ring: hue ~80-100 in OpenCV's 0-179 scale.
    teal = (ring_h >= 78) & (ring_h <= 102) & (ring_s > 110) & (ring_v > 140)
    teal_ratio = _count(teal) / float(teal.size)

    _, party_v, party_s, _ = crop("party_panel")
    party_bright = _count((party_v > 150) & (party_s < 90)) / float(party_v.size)

    _, hp_v, hp_s, hp_h = crop("enemy_hp")
    hp_bars = _count((hp_v > 120) & ((hp_h < 12) | (hp_h > 160)) & (hp_s > 90)) / float(hp_v.size)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = _count(edges) / float(edges.size)

    return Features(
        mean_brightness=float(value.mean()),
        dialogue_band_dark=dark_band,
        dialogue_text_pixels=text_pixels,
        teal_ring_pixels=teal_ratio,
        party_panel_bright=party_bright,
        enemy_hp_bar_pixels=hp_bars,
        edge_density=edge_density,
    )
