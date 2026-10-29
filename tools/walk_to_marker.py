"""Walk to the quest marker using the world pip and the minimap together.

Two markers, two jobs, and one measurement that decides how they may be combined:

  world pip  (gold diamond with an "Nm" label, in the 3D scene)
             a *camera-relative* direction: when it is on screen, its distance from
             the screen centre is a fine steering error, and no calibration is needed.
  minimap    (gold diamond on the disc, cyan dart for the player)
             a *world* direction: the dart is the character's facing and the disc does
             not rotate when the camera free-looks (measured: best-match rotation 0.0
             deg, beating +-1 deg), so the pip offset is a bearing in the world, not a
             camera rotation.

    So the minimap cannot say "turn the camera this much": the camera's yaw is a
    separate unknown. What it *can* say is "the character is facing this far off the
    objective", and the character's facing is directly controllable -- pushing the stick
    walks her in that direction and turns her to face it. With the camera held still,
    a stick push at screen angle s makes her face world angle yaw + s, so adjusting s by
    the measured error (objective bearing - facing) converges without ever knowing yaw.

The loop therefore runs two phases, never mixed, because mixing them invalidates each
other's assumptions:

  heading  camera still, character facing != objective bearing
           -> push the stick at an angle corrected by the measured turn error
  camera   character is facing the objective but the world pip is off screen
           -> sweep the camera in fixed strokes until the pip appears
  steer    pip visible -> push toward it; stop at 1-2 m

    python tools/walk_to_marker.py --steps 40
    python tools/walk_to_marker.py --dry-run          # report, do not touch the game
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.game_hsr import minimap  # noqa: E402
from adapters.game_hsr.perception import RapidOcr  # noqa: E402

DISTANCE_RE = re.compile(r"^(\d{1,3})\s*m$", re.IGNORECASE)
STICK = (250, 570)
SCREEN_CENTRE_X = 640

HEADING_TOL_DEG = 25.0     # closer than this and the character is heading at the pip
MAX_SCREEN_PUSH_DEG = 45.0  # a pip at the screen edge means "push this far off axis"
STICK_MAGNITUDE_PX = 170
STICK_DURATION_MS = 750
CAMERA_STROKE_PX = 260
CAMERA_SWEEP_LIMIT = 14    # strokes before reversing; 14 x 260 px covers a full turn
OCCLUSION_TOLERANCE = 2    # frames of a hidden pip the walker rides out before sweeping
FACING_CONF_MIN = 0.4      # below this the dart's shape is not trustworthy


def fetch(base: str):
    with urllib.request.urlopen(f"{base}/frame", timeout=10) as response:
        return cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)


def act(base: str, body: dict) -> None:
    request = urllib.request.Request(
        f"{base}/action", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    urllib.request.urlopen(request, timeout=15).read()


def push(base: str, dx: int, dy: int, duration_ms: int = STICK_DURATION_MS) -> None:
    act(base, {"type": "swipe", "x": STICK[0], "y": STICK[1],
               "x2": STICK[0] + dx, "y2": STICK[1] + dy, "duration_ms": duration_ms})


def push_at(base: str, angle_deg: float, magnitude_px: int = STICK_MAGNITUDE_PX,
            duration_ms: int = STICK_DURATION_MS) -> None:
    """Push the stick at a screen angle: 0 = up, clockwise, like the minimap angles."""
    radians = math.radians(angle_deg)
    push(base, int(round(magnitude_px * math.sin(radians))),
         int(round(-magnitude_px * math.cos(radians))), duration_ms)


def rotate(base: str, direction: int, magnitude: int = CAMERA_STROKE_PX) -> None:
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


def wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk to the quest marker")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--stop-m", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("data/phase1"))
    parser.add_argument("--dry-run", action="store_true",
                        help="report what the loop sees without sending any input")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    ocr = RapidOcr()
    args.out.mkdir(parents=True, exist_ok=True)

    previous_gray = None
    stuck_steps = 0
    stick_angle = 0.0        # screen angle the stick is being pushed at
    sweeps_in_direction = 0
    sweep_direction = 1
    missing_pip = OCCLUSION_TOLERANCE + 1  # nothing seen yet, so a sweep is fair game

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

        view = minimap.read(frame)
        pip = world_pip(ocr, frame)
        # Near the objective the pip and the arrow start to overlap on the disc, which
        # corrupts the dart's shape; a low-confidence facing is worse than none, because
        # it turns into a confident push in the wrong direction.
        heading_error = view.turn_deg if view.facing_confidence >= FACING_CONF_MIN else None

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
                if not args.dry_run:
                    push(base, 0, 80, 900)
                    time.sleep(0.2)
                    rotate(base, sweep_direction)
                stuck_steps = 0
                print(f"step {step:02d}: stuck at {metres}m; backed off + rotated")
                time.sleep(0.3)
                continue

            error = pip_x - SCREEN_CENTRE_X
            # Proportional steering: a pip 265 px left of centre is ~19 deg of camera
            # yaw, and pushing the stick at a flat 45 deg makes her arc past it -- the
            # objective then sits on the other side of her heading and the error stops
            # shrinking. Scale the stick angle to the error instead.
            stick_angle = wrap(error / SCREEN_CENTRE_X * MAX_SCREEN_PUSH_DEG)
            if not args.dry_run:
                push_at(base, stick_angle, magnitude_px=150)
            print(f"step {step:02d}: pip {metres:3d}m x={pip_x:4d} err={error:+5d} "
                  f"-> stick {stick_angle:+6.1f} | facing={view.facing_deg} "
                  f"bearing={view.bearing_deg}")
            time.sleep(0.3)
            sweeps_in_direction = 0
            missing_pip = 0
            continue

        # No world pip. The screen was the better measurement while it lasted, so keep
        # walking the way it said for a couple of steps before trusting the minimap: a
        # pip vanishing is usually a pillar, and the minimap can be reading a different
        # gold diamond entirely (chests and anchors share the colour).
        if missing_pip <= OCCLUSION_TOLERANCE:
            if not args.dry_run:
                push_at(base, stick_angle, magnitude_px=150)
            print(f"step {step:02d}: pip briefly hidden ({missing_pip}); "
                  f"holding stick {stick_angle:+6.1f}")
            missing_pip += 1
            time.sleep(0.3)
            continue

        if heading_error is not None and abs(heading_error) > HEADING_TOL_DEG:
            stick_angle = wrap(stick_angle + heading_error)
            if not args.dry_run:
                push_at(base, stick_angle)
            print(f"step {step:02d}: heading {heading_error:+6.1f} deg "
                  f"(facing {view.facing_deg:5.1f}, objective {view.bearing_deg:5.1f})"
                  f" -> stick {stick_angle:+6.1f}")
            time.sleep(0.3)
            continue

        # Facing the objective, pip off screen: the camera is what needs moving.
        if sweeps_in_direction >= CAMERA_SWEEP_LIMIT:
            sweep_direction *= -1
            sweeps_in_direction = 0
            print(f"step {step:02d}: full sweep found no pip; reversing")
        if not args.dry_run:
            rotate(base, sweep_direction)
        sweeps_in_direction += 1
        print(f"step {step:02d}: pip off-screen, heading aligned "
              f"(facing {view.facing_deg}, objective {view.bearing_deg}); "
              f"camera sweep {sweep_direction:+d} ({sweeps_in_direction})")
        time.sleep(0.35)

    cv2.imwrite(str(args.out / "walker-final.jpg"), fetch(base))
    print("steps exhausted; saved walker-final.jpg")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
