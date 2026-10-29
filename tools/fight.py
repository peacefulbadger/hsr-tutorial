"""Attack the selected enemy until combat ends, stopping on anything that needs thought.

The turn-based fight itself is not interesting to reason about: the skills are on a fixed
layout and the tutorial only ever asks for one of them, so a loop that presses the active
action and notices when the fight is over is enough. What matters is the *exit*: the loop
must stop on a reward screen, a blocking dialogue, or a dialogue with choices, because
those are the states where a blind press either skips story or picks an answer.

    python tools/fight.py --seconds 240
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

from adapters.game_hsr.perception import HsrPerception, RapidOcr  # noqa: E402

# The large action button bottom-right: basic attack, and the skill when it replaces it.
# This layout is the 1280x720 one measured off captured frames; the button sits slightly
# lower and further right than the naive "bottom right corner" guess.
ACTION_BUTTON = (1045, 590)
# Ultimate sits on the character portrait, bottom-left, and only lights up at full energy.
ULTIMATE_BUTTON = (230, 590)
# Continue affordances: reward screens and dialogues advance on a click near the centre.
CONTINUE_POINT = (640, 400)
STOP_ON = {"dialogue", "dialogue_choice", "reward"}
# Combat frames are dimmed by tutorial overlays often enough that the classifier reports
# "unknown" while a fight is plainly running. Tapping through "unknown" is safe -- during
# an enemy turn the press is ignored -- and stopping there would stall the fight forever.
TAP_THROUGH = {"combat", "unknown"}


def tutorial_request(boxes) -> str | None:
    """What the tutorial banner is asking for, when it is asking for something.

    The tutorial gates input: it dims the UI and waits for one specific press, so an
    attack tap into that state is discarded and the fight silently stops progressing.
    Its banner says which press it wants in plain words, which is the only reliable
    signal -- the button it highlights is drawn differently every time it changes.
    """
    for box in boxes:
        text = box.text.lower()
        if "activate your ultimate" in text or "ultimate now" in text:
            return "ultimate"
    return None


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
    parser = argparse.ArgumentParser(description="Fight until the game asks for input")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--seconds", type=float, default=240.0)
    parser.add_argument("--interval", type=float, default=1.1)
    parser.add_argument("--out", type=Path, default=Path("data/phase1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=True)
    perception = HsrPerception(RapidOcr())

    deadline = time.monotonic() + args.seconds
    last = ""
    seen_combat = False
    while time.monotonic() < deadline:
        frame = fetch(base)
        screen, confidence, _features, boxes = perception.analyze(frame)
        screen = "dialogue" if screen == "dialogue_bark" else screen

        if screen != last:
            print(f"{time.strftime('%H:%M:%S')} {screen} ({confidence:.2f})")
            last = screen

        if screen in TAP_THROUGH:
            seen_combat = True
            request = tutorial_request(boxes)
            point = ULTIMATE_BUTTON if request == "ultimate" else ACTION_BUTTON
            if request is not None:
                print(f"{time.strftime('%H:%M:%S')} tutorial asks for {request} -> tap {point}")
            if not args.dry_run:
                act(base, {"type": "tap", "x": point[0], "y": point[1]})
            time.sleep(args.interval)
            continue

        if screen in STOP_ON and seen_combat:
            cv2.imwrite(str(args.out / f"fight-stopped-{screen}.jpg"), frame)
            print(f"stopped on {screen}; saved fight-stopped-{screen}.jpg")
            return 0

        if screen == "reward":
            if not args.dry_run:
                act(base, {"type": "tap", "x": CONTINUE_POINT[0], "y": CONTINUE_POINT[1]})
            time.sleep(args.interval)
            continue

        # Loading, transition, menus between rounds: just wait for the next state.
        time.sleep(args.interval)

    print("time budget exhausted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
