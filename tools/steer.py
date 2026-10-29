"""Steer toward the quest marker using the minimap.

Blind swipe bursts just make the character orbit a plant. The minimap top-left is
a compass: the gold quest diamond sits at an offset from the player at its centre,
so that offset gives a direction every frame.

    python tools/steer.py --steps 25

Stops early if a dialogue band appears (that is what we are trying to reach).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Minimap geometry, read off captured frames: circle centred here, this radius.
MINIMAP_CENTER = (100, 130)
MINIMAP_RADIUS = 68

# The joystick; swipes start here and push in a direction.
STICK = (250, 570)
PUSH = 140


def fetch(base: str):
    with urllib.request.urlopen(f"{base}/frame", timeout=10) as response:
        body = response.read()
        headers = response.headers
    frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    return frame, headers


def swipe(base: str, dx: int, dy: int, duration_ms: int = 1100) -> dict:
    payload = json.dumps({
        "type": "swipe",
        "x": STICK[0], "y": STICK[1],
        "x2": STICK[0] + dx, "y2": STICK[1] + dy,
        "duration_ms": duration_ms,
    }).encode()
    request = urllib.request.Request(
        f"{base}/action", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read())


def marker_offset(frame) -> tuple[float, float] | None:
    """Gold quest diamond inside the minimap, relative to the player centre."""
    cx, cy = MINIMAP_CENTER
    r = MINIMAP_RADIUS
    crop = frame[cy - r:cy + r, cx - r:cx + r]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    gold = (hue >= 15) & (hue <= 40) & (sat > 140) & (val > 180)

    ys, xs = np.nonzero(gold)
    if len(xs) < 8:
        return None
    mask = (xs - r) ** 2 + (ys - r) ** 2 <= (r - 4) ** 2
    if mask.sum() < 8:
        return None
    mx = float(xs[mask].mean()) - r
    my = float(ys[mask].mean()) - r
    return mx, my


def dialogue_visible(frame) -> bool:
    """Cheap check for the subtitle band: dark strip, bright glyphs."""
    band = frame[600:700, 200:1080]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    bright_text = ((val > 165) & (sat < 70)).mean()
    return bright_text > 0.004


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Steer toward the quest marker")
    parser.add_argument("--url", default="http://10.0.0.1:8765")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("data/phase1"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip("/")

    for step in range(1, args.steps + 1):
        frame, _headers = fetch(base)
        if dialogue_visible(frame):
            path = args.out / f"steer-dialogue-{step:02d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"step {step:02d}: dialogue detected -> {path}")
            return 0

        offset = marker_offset(frame)
        if offset is None:
            dx, dy = 0, -PUSH  # no marker found: keep walking forward
            direction = "forward (no marker)"
        else:
            mx, my = offset
            # Push toward the marker. Screen y is inverted relative to stick y.
            scale = min(1.0, 40.0 / max(abs(mx), abs(my), 1.0))
            dx = int(max(-PUSH, min(PUSH, mx * 3 * scale)))
            dy = int(max(-PUSH, min(PUSH, my * 3 * scale)))
            if abs(dy) < 30:
                dy = -60 if abs(mx) < 20 else 0
            direction = f"marker dx={mx:+.0f} dy={my:+.0f}"

        swipe(base, dx, dy)
        time.sleep(0.35)
        print(f"step {step:02d}: {direction} -> swipe({dx:+d},{dy:+d})")

    frame, _headers = fetch(base)
    cv2.imwrite(str(args.out / "steer-final.jpg"), frame)
    print("steps exhausted; saved steer-final.jpg")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
