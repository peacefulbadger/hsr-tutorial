"""Walk toward the quest marker, using its own distance label as feedback.

The marker renders a "4m" label beside a gold diamond. That label is a better
controller input than the minimap (whose pip is decoration) and better than
dead reckoning, because it tells us both direction and whether we are getting
closer.

    python tools/walk_to_marker.py --steps 30

Control loop, one step per iteration:
  - find the "Nm" label (OCR)
  - if no label: sweep the camera, the marker may be behind us
  - if distance <= stop_m: we have arrived
  - else push the stick toward the label's side of the screen
  - if the screen stopped changing for two consecutive steps, we are against
    geometry: back off and rotate instead of grinding into the wall
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.game_hsr.perception import RapidOcr  # noqa: E402

DISTANCE_RE = re.compile(r"^(\d{1,3})\s*m$", re.IGNORECASE)
STICK = (250, 570)


def fetch(base: str):
    with urllib.request.urlopen(f"{base}/frame", timeout=10) as response:
        return cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)


def act(base: str, body: dict) -> None:
    request = urllib.request.Request(
        f"{base}/action", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    urllib.request.urlopen(request, timeout=15).read()


def push(base: str, dx: int, dy: int, duration_ms: int = 1300) -> None:
    act(base, {"type": "swipe", "x": STICK[0], "y": STICK[1],
               "x2": STICK[0] + dx, "y2": STICK[1] + dy, "duration_ms": duration_ms})


def rotate(base: str, dx: int = -380) -> None:
    act(base, {"type": "swipe", "x": 950, "y": 360, "x2": 950 + dx, "y2": 360,
               "duration_ms": 700})


def find_marker(ocr, frame) -> tuple[int, int, int] | None:
    """Return (centre_x, centre_y, metres) for the quest distance label."""
    best = None
    for box in ocr.read(frame):
        match = DISTANCE_RE.match(box.text.replace(" ", ""))
        if not match:
            continue
        metres = int(match.group(1))
        x, y, w, h = box.bbox
        best = (x + w // 2, y + h // 2, metres)
        break
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk to the quest marker")
    parser.add_argument("--url", default="http://10.0.0.1:8765")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--stop-m", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("data/phase1"))
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    ocr = RapidOcr()
    args.out.mkdir(parents=True, exist_ok=True)

    previous_gray = None
    stuck_steps = 0

    for step in range(1, args.steps + 1):
        frame = fetch(base)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = 0.0
        if previous_gray is not None:
            motion = float(np.abs(gray.astype(np.int16) - previous_gray.astype(np.int16)).mean())
        previous_gray = gray

        hud = frame[500:660, 990:1260]
        hsv = cv2.cvtColor(hud, cv2.COLOR_BGR2HSV)
        hud_ratio = float(((hsv[:, :, 2] > 205) & (hsv[:, :, 1] < 45)).mean())
        if hud_ratio < 0.02 and gray.mean() > 20:
            cv2.imwrite(str(args.out / "walker-dialogue.jpg"), frame)
            print(f"step {step:02d}: movement HUD gone -> dialogue")
            return 0

        marker = find_marker(ocr, frame)
        if marker is None:
            rotate(base)
            print(f"step {step:02d}: no marker label; rotated camera")
            time.sleep(0.3)
            continue

        mx, my, metres = marker
        if metres <= args.stop_m:
            cv2.imwrite(str(args.out / "walker-arrived.jpg"), frame)
            print(f"step {step:02d}: arrived ({metres}m)")
            return 0

        if motion < 3.0:
            stuck_steps += 1
        else:
            stuck_steps = 0

        if stuck_steps >= 2:
            push(base, 0, 80, 900)   # back away
            time.sleep(0.2)
            rotate(base)
            stuck_steps = 0
            print(f"step {step:02d}: stuck at {metres}m; backed off and rotated")
            time.sleep(0.3)
            continue

        if mx < 480:
            push(base, -140, -100)
            direction = "left"
        elif mx > 800:
            push(base, 140, -100)
            direction = "right"
        else:
            push(base, 0, -140)
            direction = "forward"
        print(f"step {step:02d}: {metres:3d}m marker@({mx:4d},{my:3d}) -> {direction}")
        time.sleep(0.3)

    cv2.imwrite(str(args.out / "walker-final.jpg"), fetch(base))
    print("steps exhausted; saved walker-final.jpg")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
