"""Dialogue options: which stacked lines on screen are answers the player must pick.

A choice screen is not a dialogue with extra text. It is the one screen where a wrong
press is unrecoverable -- the game takes the option under the tap and moves on -- so the
detector has to be conservative, and it has to be wrong in the *quiet* direction.

Three impostors all look like a stack of short lines at first glance, and all three were
seen on the live client before this module existed:

  a name plate above a subtitle   "Kafka" / "destination?" -- 78 px apart, looks like two
                                  options until you notice the heights differ (33 vs 20)
  a wrapped subtitle              two lines of one sentence -- ~28 px apart, too tight
  quest tracker lines             "Meet up with Silver Wolf and leave the / Reception
                                  Center" -- left edge, but inside the subtitle band

So an option line must: sit in the choice region, share a left edge, share a font height
(they are rendered identically), be spaced like separate rows rather than wrapped lines,
and not be part of the subtitle band at all.
"""

from __future__ import annotations

from dataclasses import dataclass

# Where options appear: below the speaker, left of the character. Generous on purpose --
# the group filter below is what rejects impostors, not this box.
CHOICE_REGION = (380, 250, 800, 380)
# A subtitle sits here; anything inside it is the spoken line, never an option.
SUBTITLE_REGION = (150, 520, 980, 190)

MIN_OPTION_GAP = 40     # wrapped lines of one sentence are ~28 px apart
MAX_OPTION_GAP = 130
ALIGN_TOLERANCE = 60    # options share a left edge
MAX_HEIGHT_SPREAD = 8   # one font, so one height


@dataclass(frozen=True)
class Option:
    text: str
    bbox: list[int]
    confidence: float

    @property
    def centre(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


def _centre_inside(bbox: list[int], region: tuple[int, int, int, int]) -> bool:
    x, y, w, h = bbox
    rx, ry, rw, rh = region
    cx, cy = x + w / 2.0, y + h / 2.0
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def find_options(boxes, *, region: tuple[int, int, int, int] | None = CHOICE_REGION,
                 exclude: tuple[int, int, int, int] | None = SUBTITLE_REGION,
                 crop_offset: tuple[int, int] = (0, 0),
                 min_gap: int = MIN_OPTION_GAP,
                 max_gap: int = MAX_OPTION_GAP) -> list[Option]:
    """Return the option stack, top to bottom, or an empty list.

    `boxes` are OCR results; `crop_offset` shifts them into frame coordinates when the
    caller OCR'd a sub-image instead of the whole frame.
    """
    ox, oy = crop_offset
    candidates: list[Option] = []
    for box in boxes:
        x, y, w, h = box.bbox
        if w < 60 or h < 12:            # too small to be an option line
            continue
        text = box.text.strip()
        if len(text) < 2 or len(text) > 60:
            continue
        absolute = [x + ox, y + oy, w, h]
        if region is not None and not _centre_inside(absolute, region):
            continue
        if exclude is not None and _centre_inside(absolute, exclude):
            continue
        candidates.append(Option(text=text, bbox=absolute,
                                 confidence=round(float(box.confidence), 3)))

    # Group by x alignment: options share a left edge.
    groups: list[list[Option]] = []
    for option in sorted(candidates, key=lambda o: o.bbox[1]):
        for group in groups:
            if abs(group[-1].bbox[0] - option.bbox[0]) <= ALIGN_TOLERANCE:
                group.append(option)
                break
        else:
            groups.append([option])

    for group in groups:
        if len(group) < 2:
            continue
        heights = [option.bbox[3] for option in group]
        if max(heights) - min(heights) > MAX_HEIGHT_SPREAD:
            continue
        gaps = [group[i + 1].bbox[1] - group[i].bbox[1] for i in range(len(group) - 1)]
        if all(min_gap <= gap <= max_gap for gap in gaps):
            return group
    return []
