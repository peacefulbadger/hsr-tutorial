"""Watch the live client for a dialogue-choice screen and grab it.

Choices render as a short column of option lines in the middle of the screen: a
few boxes, similar left edge, stacked vertically, each a phrase rather than a
sentence. That shape is distinctive enough to detect with OCR alone.

    python tools/watch_choice.py --minutes 15

On detection it writes the frames either side of the transition (so the moments
before the choice are not lost) plus a JSON summary, then exits 0. Keep it
running while a human plays; it does not send any input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.game_hsr.perception import RapidOcr  # noqa: E402

# Where dialogue options appear. Below the speaker, left of the character.
CHOICE_REGION = (380, 250, 800, 380)
# A subtitle sits here; if one is present this is dialogue, not a choice.
SUBTITLE_REGION = (150, 520, 980, 190)

# Options are separated by a clear gap. A wrapped sentence uses normal line
# spacing (~29 px), which is what produced the first false positive.
MIN_OPTION_GAP = 40
MAX_OPTION_GAP = 130


def fetch(base: str):
    import urllib.request

    with urllib.request.urlopen(f"{base}/frame", timeout=10) as response:
        return np.frombuffer(response.read(), np.uint8)


def looks_like_choices(boxes, crop_x: int, crop_y: int, min_gap: int = MIN_OPTION_GAP,
                       exclude: tuple[int, int, int, int] | None = SUBTITLE_REGION) -> list[dict]:
    """Find a vertical stack of short option lines.

    Two things that look like a stack but are not: a name plate above a subtitle, and the
    wrapped last line of that subtitle. Both are rejected structurally -- option lines are
    all rendered in one font, so their heights match, and none of them lives inside the
    subtitle band, whereas the speaker's line always does.
    """
    candidates = []
    for box in boxes:
        x, y, w, h = box.bbox
        if w < 60 or h < 12:            # too small to be an option line
            continue
        if len(box.text) < 2 or len(box.text) > 60:
            continue
        absolute = (x + crop_x, y + crop_y, w, h)
        if exclude is not None and _centre_inside(absolute, exclude):
            continue
        candidates.append({"text": box.text, "bbox": list(absolute),
                           "confidence": round(box.confidence, 3)})

    # Group by x alignment: options share a left edge within ~60px.
    groups: list[list[dict]] = []
    for item in sorted(candidates, key=lambda c: c["bbox"][1]):
        placed = False
        for group in groups:
            if abs(group[-1]["bbox"][0] - item["bbox"][0]) <= 60:
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])

    for group in groups:
        if len(group) < 2:
            continue
        heights = [item["bbox"][3] for item in group]
        if max(heights) - min(heights) > 8:
            continue
        gaps = [group[i + 1]["bbox"][1] - group[i]["bbox"][1] for i in range(len(group) - 1)]
        if all(min_gap <= gap <= MAX_OPTION_GAP for gap in gaps):
            return group
    return []


def _centre_inside(bbox, region) -> bool:
    x, y, w, h = bbox
    rx, ry, rw, rh = region
    cx, cy = x + w / 2.0, y + h / 2.0
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch for a dialogue choice")
    parser.add_argument("--url", default="http://10.0.0.1:8765")
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--out", type=Path, default=Path("data/phase1/choice"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip("/")
    ocr = RapidOcr()
    rx, ry, rw, rh = CHOICE_REGION

    recent: deque[bytes] = deque(maxlen=6)
    deadline = time.monotonic() + args.minutes * 60
    polls = 0

    print(f"watching {base} for a choice screen, up to {args.minutes:g} min")
    while time.monotonic() < deadline:
        polls += 1
        try:
            raw = fetch(base)
        except Exception as exc:  # noqa: BLE001
            print(f"  frame fetch failed: {exc}")
            time.sleep(1.0)
            continue

        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            time.sleep(0.5)
            continue
        recent.append(raw)

        sx, sy, sw, sh = SUBTITLE_REGION
        subtitle_boxes = ocr.read(frame[sy:sy + sh, sx:sx + sw])
        subtitle_text = " ".join(b.text for b in subtitle_boxes).strip()
        # A spoken line above the options is normal on some choice screens, so
        # do not skip -- but demand more separation, because a subtitle's own
        # wrapped lines are ~29px apart and would otherwise look like options.
        min_gap = 55 if len(subtitle_text) >= 8 else MIN_OPTION_GAP

        crop = frame[ry:ry + rh, rx:rx + rw]
        options = looks_like_choices(ocr.read(crop), rx, ry, min_gap)
        if options:
            stamp = int(time.time())
            for offset, buffered in enumerate(recent):
                (args.out / f"{stamp}-pre{offset}.jpg").write_bytes(bytes(buffered))
            cv2.imwrite(str(args.out / f"{stamp}-choice.jpg"), frame)
            summary = {
                "detected_epoch_s": stamp,
                "polls": polls,
                "options": options,
                "frame": str(args.out / f"{stamp}-choice.jpg"),
            }
            (args.out / f"{stamp}-choice.json").write_text(json.dumps(summary, indent=1))
            print("CHOICE DETECTED")
            for option in options:
                print(f"  {option['text']!r} at {option['bbox']}")
            return 0

        if polls % 20 == 0:
            print(f"  {polls} polls, {int(deadline - time.monotonic())}s left")
        time.sleep(0.6)

    print("no choice screen seen before the deadline")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
