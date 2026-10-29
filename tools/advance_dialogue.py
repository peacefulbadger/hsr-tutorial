"""Advance a story dialogue, taking the first option whenever a choice appears.

The standing policy for this tutorial run is "take the top option and keep going", so this
loop is allowed to press through a choice -- but only after recording the frame and the
options it saw, and only through the same detector the rest of the project uses. A choice
is the one press that cannot be undone, so the record comes before the tap.

    python tools/advance_dialogue.py --minutes 6
    python tools/advance_dialogue.py --stop-on-choice   # record and stop, do not press
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.game_hsr.choices import find_options  # noqa: E402
from adapters.game_hsr.perception import HsrPerception, RapidOcr  # noqa: E402

# The advance affordance: low and left of centre, clear of where options render.
ADVANCE_POINT = (300, 655)
# Screens that are a conversation to press through. "unknown" covers exploration and the
# dimmed tutorial overlays, where a press is harmless.
PRESS_THROUGH = {"dialogue", "dialogue_bark", "unknown"}


def fetch(base: str) -> np.ndarray:
    with urllib.request.urlopen(f"{base}/frame", timeout=10) as response:
        return cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)


def act(base: str, body: dict) -> None:
    request = urllib.request.Request(
        f"{base}/action", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    urllib.request.urlopen(request, timeout=15).read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Advance dialogue, taking the top option")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--minutes", type=float, default=6.0)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=Path("data/phase1/choice"))
    parser.add_argument("--stop-on-choice", action="store_true",
                        help="record the choice and stop instead of taking the top option")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=True)
    perception = HsrPerception(RapidOcr())

    deadline = time.monotonic() + args.minutes * 60
    presses = 0
    taken = 0

    while time.monotonic() < deadline:
        frame = fetch(base)
        screen, confidence, _features, boxes = perception.analyze(frame)

        if screen == "dialogue_choice":
            options = find_options(boxes)
            if len(options) >= 2:
                stamp = int(time.time())
                # Record before pressing: if the tap lands wrong, the decision is still on
                # disk with the options and their coordinates.
                cv2.imwrite(str(args.out / f"{stamp}-choice.jpg"), frame)
                (args.out / f"{stamp}-choice.json").write_text(json.dumps({
                    "detected_epoch_s": stamp,
                    "screen_type": screen,
                    "options": [{"text": o.text, "bbox": o.bbox} for o in options],
                    "taken": None if args.stop_on_choice else options[0].text,
                    "presses_before": presses,
                }, indent=1))
                print("CHOICE DETECTED")
                for index, option in enumerate(options):
                    print(f"  [{index}] {option.text!r} at {option.bbox}")
                if args.stop_on_choice:
                    print("  recorded only (--stop-on-choice)")
                    return 0
                x, y = options[0].centre
                if not args.dry_run:
                    act(base, {"type": "tap", "x": x, "y": y})
                taken += 1
                print(f"  took option [0] {options[0].text!r} at ({x},{y}) "
                      f"-> saved {stamp}-choice.jpg")
                time.sleep(args.interval)
                continue

        if screen in PRESS_THROUGH:
            if not args.dry_run:
                act(base, {"type": "tap", "x": ADVANCE_POINT[0], "y": ADVANCE_POINT[1]})
            presses += 1
            if presses % 10 == 0:
                print(f"{presses} presses | {screen} ({confidence:.2f}) | choices taken {taken}")
            time.sleep(args.interval)
            continue

        print(f"stopped: screen is {screen} ({confidence:.2f}); "
              f"{presses} presses, {taken} choices taken")
        cv2.imwrite(str(args.out / f"stopped-{screen}.jpg"), frame)
        return 1

    print(f"time budget exhausted ({presses} presses, {taken} choices taken)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
