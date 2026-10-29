"""Play the tutorial: hand each screen to the loop that knows how to handle it.

Every part of this exists already -- walking, fighting, advancing dialogue, taking the top
option -- but they were being driven by hand, one story beat per decision. This is the
dispatcher: look at the screen, run the matching sub-loop for a bounded stretch, look
again. What it buys is unattended progress, and a single place where "what state am I in"
is decided.

The state decision is the whole content of this file, so it is spelled out rather than
inferred from the classifier alone:

  combat      the teal action ring, or any combat keyword, or a turn banner
  defeat      a defeat panel (the fight loop retries it, per the standing policy)
  choice      an option stack -- recorded, then option 0 is taken
  dialogue    a blocking conversation with no options
  reward      a result screen to dismiss
  everything else is the world, and the walker drives toward the quest pip

    python tools/play.py --minutes 20
    python tools/play.py --minutes 20 --dry-run     # decide, print, do not act
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import advance_dialogue  # noqa: E402
import fight  # noqa: E402
import walk_to_marker  # noqa: E402
from adapters.game_hsr.perception import HsrPerception, RapidOcr  # noqa: E402

COMBAT_KEYWORDS = ("yourturn", "enemysturn", "totaldamage", "voidranger", "weaknessbreak",
                   "basicatk", "singletarget", "blast")
WALK_STEPS = 12


def fetch(base: str) -> np.ndarray:
    with urllib.request.urlopen(f"{base}/frame", timeout=10) as response:
        return cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)


def classify_state(frame, perception) -> str:
    screen, _confidence, features, boxes = perception.analyze(frame)
    compact = "".join(ch for ch in " ".join(b.text for b in boxes).lower() if ch.isalnum())

    if fight.looks_like_defeat(boxes):
        return "defeat"
    if screen == "dialogue_choice":
        return "choice"
    if screen == "reward":
        return "reward"
    if features.teal_ring_pixels > 0.04 or any(word in compact for word in COMBAT_KEYWORDS):
        return "combat"
    if screen in {"dialogue", "dialogue_bark"}:
        return "dialogue"
    return "world"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play the tutorial end to end")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=Path("data/phase1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=True)
    perception = HsrPerception(RapidOcr())
    deadline = time.monotonic() + args.minutes * 60
    rounds = 0

    while time.monotonic() < deadline:
        rounds += 1
        frame = fetch(base)
        state = classify_state(frame, perception)
        remaining = int(deadline - time.monotonic())
        print(f"[{time.strftime('%H:%M:%S')}] round {rounds}: {state} "
              f"({remaining}s left)", flush=True)

        if args.dry_run:
            time.sleep(1.0)
            continue

        if state in {"combat", "defeat"}:
            fight.main(["--url", base, "--seconds", "150"])
        elif state in {"choice", "dialogue", "reward"}:
            advance_dialogue.main(["--url", base, "--minutes", "2"])
        else:
            walk_to_marker.main(["--url", base, "--steps", str(WALK_STEPS)])

    cv2.imwrite(str(args.out / "play-final.jpg"), fetch(base))
    print(f"time budget exhausted after {rounds} rounds; saved play-final.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
