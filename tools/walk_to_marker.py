"""Walk to the quest marker using the world pip and the minimap together.

Two markers, two jobs:

  world pip  (gold diamond with an "Nm" label, in the 3D scene)
             moves with character position and camera orientation, so its screen
             x is a fine steering error once it is visible.
  minimap pip (gold diamond inside the minimap, player is the cyan arrow)
             coarse: when the world pip is off-screen, the offset here says which
             way to rotate the camera to bring it back.

Control loop:
  - read the world pip's "Nm" label -> metres + screen x
  - read the minimap -> bearing to the objective relative to the player
  - pip visible  -> steer so the pip sits near screen centre; stop at 1-2 m
  - pip missing  -> rotate toward the bearing until it appears
  - no pixel motion twice in a row -> back off and rotate (wall contact)

    python tools/walk_to_marker.py --steps 40
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
SCREEN_CENTRE_X = 640

# Minimap: circle centre and radius, measured off captured frames.
MINIMAP_CENTRE = (105, 130)
MINIMAP_RADIUS = 70


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


def rotate(base: str, direction: int, magnitude: int = 320) -> None:
    """Drag on the right half; direction +1 / -1 chooses which way to sweep."""
    centre = 950
    act(base, {"type": "swipe", "x": centre, "y": 360,
               "x2": centre + direction * magnitude, "y2": 360, "duration_ms": 650})


def world_pip(ocr, frame) -> tuple[int, int, int] | None:
    """The 3D objective marker, located by its distance label."""
    for box in ocr.read(frame):
        text = box.text.replace(" ", "")
        match = DISTANCE_RE.match(text)
        if not match:
            continue
        x, y, w, h = box.bbox
        # The label sits under the diamond; report the label centre.
        return x + w // 2, y + h // 2, int(match.group(1))
    return None


def minimap_bearing(frame) -> tuple[float, float] | None:
    """Objective offset from the player arrow, inside the minimap."""
    cx, cy = MINIMAP_CENTRE
    r = MINIMAP_RADIUS
    crop = frame[cy - r:cy + r, cx - r:cx + r]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    gold = (hue >= 15) & (hue <= 40) & (sat > 140) & (val > 180)
    # The player arrow is cyan/blue.
    cyan = (hue >= 85) & (hue <= 110) & (sat > 120) & (val > 160)

    ys, xs = np.nonzero(gold)
    py, px = np.nonzero(cyan)
    if len(xs) < 6 or len(px) < 6:
        return None
    return float(xs.mean() - px.mean()), float(ys.mean() - py.mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk to the quest marker")
    parser.add_argument("--url", default="http://10.0.0.1:8765")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--stop-m", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("data/phase1"))
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    ocr = RapidOcr()
    args.out.mkdir(parents=True, exist_ok=True)

    previous_gray = None
    stuck_steps = 0
    turn_sign = 1  # flips if the first sweep moves the pip the wrong way
    flips = 0

    for step in range(1, args.steps + 1):
        frame = fetch(base)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = 0.0 if previous_gray is None else float(
            np.abs(gray.astype(np.int16) - previous_gray.astype(np.int16)).mean()
        )
        previous_gray = gray

        hud = cv2.cvtColor(frame[500:660, 990:1260], cv2.COLOR_BGR2HSV)
        hud_ratio = float(((hud[:, :, 2] > 205) & (hud[:, :, 1] < 45)).mean())
        if hud_ratio < 0.02 and gray.mean() > 20:
            cv2.imwrite(str(args.out / "walker-dialogue.jpg"), frame)
            print(f"step {step:02d}: movement HUD gone -> dialogue/cutscene")
            return 0

        pip = world_pip(ocr, frame)
        bearing = minimap_bearing(frame)

        if pip is not None:
            pip_x, pip_y, metres = pip
            if metres <= args.stop_m:
                cv2.imwrite(str(args.out / "walker-arrived.jpg"), frame)
                print(f"step {step:02d}: arrived ({metres}m) pip@({pip_x},{pip_y})")
                return 0

            if motion < 3.0:
                stuck_steps += 1
            else:
                stuck_steps = 0
            if stuck_steps >= 2:
                push(base, 0, 80, 900)
                time.sleep(0.2)
                rotate(base, turn_sign)
                stuck_steps = 0
                print(f"step {step:02d}: stuck at {metres}m; backed off + rotated")
                time.sleep(0.3)
                continue

            error = pip_x - SCREEN_CENTRE_X
            if error < -140:
                push(base, -150, -110)
                direction = "left"
            elif error > 140:
                push(base, 150, -110)
                direction = "right"
            else:
                push(base, 0, -150)
                direction = "forward"
            print(f"step {step:02d}: pip {metres:3d}m x={pip_x:4d} err={error:+5d} -> {direction}")
            time.sleep(0.3)
            continue

        # Pip is off-screen: use the minimap to choose a rotation direction.
        if bearing is not None and abs(bearing[0]) > 3:
            direction = turn_sign if bearing[0] > 0 else -turn_sign
        else:
            direction = turn_sign
        rotate(base, direction)
        if flips < 2 and step > 2:
            flips += 1
        print(f"step {step:02d}: pip off-screen; bearing={bearing}; rotated {direction:+d}")
        time.sleep(0.35)

    cv2.imwrite(str(args.out / "walker-final.jpg"), fetch(base))
    print("steps exhausted; saved walker-final.jpg")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
