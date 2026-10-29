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

from adapters.game_hsr import minimap, navigation  # noqa: E402
from adapters.game_hsr.perception import HsrPerception, RapidOcr  # noqa: E402

DISTANCE_RE = re.compile(r"^(\d{1,3})\s*m$", re.IGNORECASE)
STICK = (250, 570)
SCREEN_CENTRE_X = 640

HEADING_TOL_DEG = 25.0     # closer than this and the character is heading at the pip
MAX_HEADING_STEP_DEG = 40.0  # a stick push never turns her more than this in one go
ROUTE_GAIN = 0.7           # fraction of the heading error applied per push
ROUTE_DEADBAND_DEG = 12.0  # below this, pushing on is already right; do not re-aim
ROUTE_SWITCH_CONFIRM = 2   # frames a new route bearing must survive before it is taken
MAX_SCREEN_PUSH_DEG = 45.0  # a pip at the screen edge means "push this far off axis"
STICK_MAGNITUDE_PX = 170
STICK_DURATION_MS = 750
CAMERA_STROKE_PX = 260
CAMERA_SWEEP_LIMIT = 14    # strokes before reversing; 14 x 260 px covers a full turn
OCCLUSION_TOLERANCE = 2    # frames of a hidden pip the walker rides out before sweeping
LOCK_TOLERANCE_PX = 90     # how far a locked marker may slide before the lock is dropped
FACING_CONF_MIN = 0.4      # below this the dart's shape is not trustworthy

# A stall is the only evidence that something solid is in front, and on its own it says
# nothing about where the way round is. So the decision is held for a few steps instead of
# being re-made every frame -- re-deciding is what makes the character alternate left,
# right, left against the same desk and never leave it.
DETOUR_HOLD_STEPS = 5
DETOUR_BASE_DEG = 55.0     # first sidestep: enough to clear a desk, not a full turn
DETOUR_GROWTH_DEG = 25.0   # each time that fails, bend further
DETOUR_MAX_DEG = 120.0
DETOUR_SIDE_FLIP = 3       # stalls before the other way round is tried
DETOUR_PROGRESS_M = 1      # a metre closer cancels the detour: it worked


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


def world_pips(ocr, frame) -> list[tuple[int, int, int]]:
    """Every labelled 3D objective marker on screen: (x, y, metres), left to right.

    There is usually more than one. A quest like "investigating the terminals 0/2" puts a
    labelled diamond on each terminal, and chasing "whichever the OCR returned first"
    makes the walker oscillate between two valid targets without reaching either.
    """
    pips: list[tuple[int, int, int]] = []
    for box in ocr.read(frame):
        text = box.text.replace(" ", "")
        match = DISTANCE_RE.match(text)
        if not match:
            continue
        x, y, w, h = box.bbox
        # The label sits under the diamond; report the label centre.
        pips.append((x + w // 2, y + h // 2, int(match.group(1))))
    return sorted(pips, key=lambda pip: pip[0])


def choose_pip(pips: list[tuple[int, int, int]],
               locked_x: int | None) -> tuple[tuple[int, int, int] | None, int | None]:
    """Pick one target and stay with it: the lock is what stops the thrash.

    A locked target is kept while a candidate sits within LOCK_TOLERANCE_PX of where it
    was, because the marker slides as the camera turns but not by hundreds of pixels.
    """
    if not pips:
        return None, None
    if locked_x is not None:
        nearest = min(pips, key=lambda pip: abs(pip[0] - locked_x))
        if abs(nearest[0] - locked_x) <= LOCK_TOLERANCE_PX:
            return nearest, nearest[0]
    # No usable lock: prefer the marker nearest the centre, which is where the active
    # objective tends to be, and where steering has the most room before it leaves view.
    chosen = min(pips, key=lambda pip: abs(pip[0] - SCREEN_CENTRE_X))
    return chosen, chosen[0]


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
    parser.add_argument("--ignore-blind", action="store_true",
                        help="experiment: turn off the arrow-footprint (A) handling")
    parser.add_argument("--ignore-bumps", action="store_true",
                        help="experiment: turn off the stall bump map (B)")
    parser.add_argument("--no-route", action="store_true",
                        help="experiment: never steer by the planned route, only by the "
                             "world pip and contact stalls")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    ocr = RapidOcr()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ignore_blind:
        # ARROW_NOSE_PX is read at call time, so this really does disable the footprint.
        navigation.ARROW_NOSE_PX = 0.0

    previous_gray = None
    stuck_steps = 0
    stick_angle = 0.0        # screen angle the stick is being pushed at
    sweeps_in_direction = 0
    sweep_direction = 1
    stalls = 0
    detour_side = 1          # +1 right, -1 left, of the pip direction
    detour_steps = 0         # steps left of the committed sidestep
    detour_angle = DETOUR_BASE_DEG
    detour_start_metres: int | None = None
    bumps = navigation.BumpMap()
    missing_pip = OCCLUSION_TOLERANCE + 1  # nothing seen yet, so a sweep is fair game
    previous_pip: tuple[int, int] | None = None
    locked_x: int | None = None
    smoothed_route: float | None = None
    route_flips = 0
    disc_bearing: float | None = None   # last accepted objective bearing, for pip choice
    last_metres: int | None = None
    near_readings = 0

    for step in range(1, args.steps + 1):
        frame = fetch(base)
        bumps.age()
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

        view = minimap.read(frame, prefer_bearing=disc_bearing)
        if view.has_pip:
            if disc_bearing is not None:
                shift = wrap(view.bearing_deg - disc_bearing)
                if abs(shift) > 90.0:
                    print(f"step {step:02d}: objective marker moved {shift:+.0f} deg "
                          f"on the disc ({view.pip_candidates} candidates); following it")
            disc_bearing = view.bearing_deg
        candidates = world_pips(ocr, frame)
        pip, locked_x = choose_pip(candidates, locked_x)
        if pip is not None and len(candidates) > 1:
            print(f"step {step:02d}: {len(candidates)} markers on screen; tracking "
                  f"the one at x={pip[0]} ({pip[2]}m)")
        # The distance label is OCR'd digits, and OCR misreads them: "79m" came back as
        # "3m" and then "1m" on consecutive frames, which reads as a teleport and then as
        # an arrival. Walking is continuous, so enforce that instead of trusting one frame.
        if pip is not None and last_metres is not None and abs(pip[2] - last_metres) > 15:
            print(f"step {step:02d}: ignoring implausible distance {pip[2]}m "
                  f"(last {last_metres}m)")
            pip = None
        # Near the objective the pip and the arrow start to overlap on the disc, which
        # corrupts the dart's shape; a low-confidence facing is worse than none, because
        # it turns into a confident push in the wrong direction.
        heading_error = view.turn_deg if view.facing_confidence >= FACING_CONF_MIN else None

        # The route is planned before steering because the white lines are the room's walls
        # and furniture. But it is *only* a fallback for when the marker cannot be seen:
        # the world pip is a direct, camera-relative measurement of where the objective is,
        # while the route runs off a disc that is re-segmented every frame, and steering by
        # it whenever it bends (which in a furnished room is always) makes the walk chase a
        # waypoint that flips sides between frames. Measured: 30 steps of that is a spin on
        # the spot. So the pip drives, contact stalls handle what the pip cannot see, and
        # the plan only takes over once the pip is out of sight.
        route = navigation.plan(frame, view, None if args.ignore_bumps else bumps)
        if route.blocked:
            print(f"step {step:02d}: no route to the marker; sweeping the camera to see "
                  f"more of the room")
            if not args.dry_run:
                rotate(base, sweep_direction)
            sweeps_in_direction += 1
            time.sleep(0.35)
            continue

        if pip is not None:
            pip_x, pip_y, metres = pip
            last_metres = metres
            if metres <= args.stop_m:
                # One reading is not an arrival: require it twice, so a single misread
                # digit cannot end the walk 60 m short.
                near_readings += 1
                if near_readings >= 2:
                    cv2.imwrite(str(args.out / "walker-arrived.jpg"), frame)
                    # Arriving is not the objective: the quest asks for something to be
                    # *done* at the marker. Standing next to it and reporting success is
                    # how a walk ends with the prompt on screen and nobody pressing it.
                    prompt = HsrPerception().interact_prompt(ocr.read(frame))
                    if prompt is None:
                        print(f"step {step:02d}: arrived ({metres}m) but nothing to "
                              f"press; prompt region empty")
                        return 2
                    if not args.dry_run:
                        x, y = prompt.center
                        act(base, {"type": "tap", "x": x, "y": y})
                    print(f"step {step:02d}: arrived ({metres}m); pressed "
                          f"{prompt.label!r} at {prompt.center}")
                    return 0
                print(f"step {step:02d}: {metres}m, confirming arrival")
                time.sleep(0.4)
                continue
            near_readings = 0

            # Getting closer cancels a detour: whatever she just did, it worked.
            if (detour_steps > 0 and detour_start_metres is not None
                    and metres <= detour_start_metres - DETOUR_PROGRESS_M):
                print(f"step {step:02d}: {detour_start_metres - metres}m closer after the "
                      f"detour; steering at the marker again")
                detour_steps = 0
                stalls = 0

            if detour_steps > 0:
                detour_steps -= 1
                previous_pip = (metres, pip_x)
                if not args.dry_run:
                    push_at(base, wrap(stick_angle + detour_side * detour_angle),
                            magnitude_px=150, duration_ms=700)
                print(f"step {step:02d}: still detouring "
                      f"{detour_angle:.0f} deg "
                      f"{'right' if detour_side > 0 else 'left'} at {metres}m "
                      f"({detour_steps} steps left)")
                time.sleep(0.3)
                continue

            # Stall from the pip, not from whole-frame motion: a scene with steam, particles
            # or an idle NPC animation keeps the frame diff above any sensible threshold
            # while the character is wedged against a wall, so the walk never noticed.
            if (previous_pip is not None
                    and abs(metres - previous_pip[0]) == 0
                    and abs(pip_x - previous_pip[1]) <= 2):
                stuck_steps += 1
            else:
                stuck_steps = 0
            previous_pip = (metres, pip_x)
            if stuck_steps >= 2:
                # Standing at the objective and not moving is usually a door or a console,
                # not a bad heading: the pip is centred and the walk is straight, so the
                # thing in the way is something to press. The prompt is named after the
                # object ("Controller", "Investigate"), so it is found structurally.
                prompt = HsrPerception().interact_prompt(ocr.read(frame))
                if prompt is not None and not args.dry_run:
                    x, y = prompt.center
                    act(base, {"type": "tap", "x": x, "y": y})
                    print(f"step {step:02d}: blocked at {metres}m; pressed "
                          f"{prompt.label!r} at ({x},{y})")
                    stuck_steps = 0
                    time.sleep(0.8)
                    continue
                # Nothing to press, so something solid is in the way -- and the disc is
                # not showing it, because it is standing under the nose of the arrow.
                # A stall is evidence, so record it where it happened and let the planner
                # route round it; then commit to one way round. Alternating sides every
                # frame is what keeps her grinding against the same desk forever.
                stalls += 1
                where = ""
                if view.facing_deg is not None and not args.ignore_bumps:
                    bump = bumps.record(view.facing_deg)
                    where = f", bump at {bump.bearing_deg:.0f} deg"
                if stalls % DETOUR_SIDE_FLIP == 0:
                    detour_side *= -1
                detour_angle = min(
                    DETOUR_BASE_DEG + DETOUR_GROWTH_DEG * (stalls // DETOUR_SIDE_FLIP),
                    DETOUR_MAX_DEG)
                detour_start_metres = metres
                detour_steps = DETOUR_HOLD_STEPS - 1
                if not args.dry_run:
                    push_at(base, wrap(stick_angle + detour_side * detour_angle),
                            magnitude_px=150, duration_ms=700)
                stuck_steps = 0
                previous_pip = None
                print(f"step {step:02d}: stuck at {metres}m{where}; detouring "
                      f"{detour_angle:.0f} deg "
                      f"{'right' if detour_side > 0 else 'left'} ({stalls})")
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

        # The marker has been out of sight long enough that holding the old push is a
        # guess. Now the plan is the only thing with an opinion, so follow it: the route
        # was built from the disc this frame and knows which way round the furniture leads
        # back to the objective.
        if route.needs_detour and view.facing_deg is not None and not args.no_route:
            # The plan steers the body, and the body turning does not turn the camera.
            # So while the plan is driving, sweep the camera too: otherwise the marker
            # stays off screen for the whole walk and the direct measurement that the
            # pip servo needs never comes back. A sweep is safe here because the body
            # heading it is being combined with is camera-independent.
            if not args.dry_run:
                rotate(base, sweep_direction)
            sweeps_in_direction += 1
            if sweeps_in_direction >= CAMERA_SWEEP_LIMIT:
                sweep_direction *= -1
                sweeps_in_direction = 0
            if smoothed_route is None:
                smoothed_route = route.bearing_deg
            else:
                delta = wrap(route.bearing_deg - smoothed_route)
                if abs(delta) > 60.0:
                    # The disc is re-segmented every frame and the way round a room can
                    # flip sides on noise. Hold the committed heading while it settles --
                    # keep walking, do not freeze -- and accept the new one if it survives
                    # a second frame.
                    route_flips += 1
                    if route_flips < ROUTE_SWITCH_CONFIRM:
                        print(f"step {step:02d}: route bearing jumped {delta:+.0f} deg; "
                              f"holding {smoothed_route:.0f} for another frame")
                        delta = 0.0
                    else:
                        print(f"step {step:02d}: route bearing moved {delta:+.0f} deg "
                              f"again; accepting it")
                        smoothed_route = route.bearing_deg
                        route_flips = 0
                else:
                    route_flips = 0
                if delta:
                    smoothed_route = wrap(smoothed_route + 0.5 * delta)
            error = wrap(smoothed_route - view.facing_deg)
            # Gain, not unit feedback: the stick angle is an integrator on a plant that
            # only reaches the commanded heading a push later, so applying the whole error
            # every step overshoots both ways and the walk becomes a spin.
            if abs(error) > ROUTE_DEADBAND_DEG:
                gain = ROUTE_GAIN * error
                correction = max(-MAX_HEADING_STEP_DEG, min(MAX_HEADING_STEP_DEG, gain))
                stick_angle = wrap(stick_angle + correction)
            if not args.dry_run:
                push_at(base, stick_angle,
                        magnitude_px=STICK_MAGNITUDE_PX if abs(error) > HEADING_TOL_DEG else 150)
            print(f"step {step:02d}: marker out of sight; route bends "
                  f"{route.detour_deg:+.0f} deg around the lines -> waypoint "
                  f"{smoothed_route:.0f}, stick {stick_angle:+.1f} "
                  f"(facing {view.facing_deg:.0f})")
            stuck_steps = 0
            time.sleep(0.3)
            continue

        if heading_error is not None and abs(heading_error) > HEADING_TOL_DEG:
            if abs(heading_error) > 120.0:
                # A "turn around completely" reading is more likely two gold diamonds on
                # one disc than a real 150 deg error -- they share a colour and the disc
                # picks the largest. Sweeping the camera reacquires the world pip instead,
                # which is a measurement rather than a guess.
                if not args.dry_run:
                    rotate(base, sweep_direction)
                sweeps_in_direction += 1
                print(f"step {step:02d}: heading error {heading_error:+.1f} deg is too "
                      f"large to trust (facing {view.facing_deg}, objective "
                      f"{view.bearing_deg}); sweeping camera instead")
                time.sleep(0.35)
                continue
            correction = max(-MAX_HEADING_STEP_DEG, min(MAX_HEADING_STEP_DEG, heading_error))
            stick_angle = wrap(stick_angle + correction)
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
