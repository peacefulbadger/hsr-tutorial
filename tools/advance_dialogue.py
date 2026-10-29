"""Advance a story dialogue, stopping the instant it turns into a choice.

Pressing "continue" blind through a conversation is fine right up until the game asks a
question: the next press would then pick whichever option happens to sit under the tap.
So the loop checks for an option stack *before* every press, and treats a detected stack
as a hard stop that saves the frame and returns.

    python tools/advance_dialogue.py --minutes 6
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.game_hsr.perception import HsrPerception, RapidOcr  # noqa: E402
from watch_choice import CHOICE_REGION, SUBTITLE_REGION, looks_like_choices  # noqa: E402

# The advance affordance: the dialogue strip runs along the bottom, so a press low and
# left of centre continues the line without touching the middle where options appear.
ADVANCE_POINT = (300, 655)


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
    parser = argparse.ArgumentParser(description="Advance dialogue, stop on a choice")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--minutes", type=float, default=6.0)
    parser.add_argument("--interval", type=float, default=1.6)
    parser.add_argument("--out", type=Path, default=Path("data/phase1/choice"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=True)
    ocr = RapidOcr()
    perception = HsrPerception(ocr)

    rx, ry, rw, rh = CHOICE_REGION
    sx, sy, sw, sh = SUBTITLE_REGION
    deadline = time.monotonic() + args.minutes * 60
    presses = 0

    while time.monotonic() < deadline:
        frame = fetch(base)

        subtitle = " ".join(
            b.text for b in ocr.read(frame[sy:sy + sh, sx:sx + sw])
        ).strip()
        min_gap = 55 if len(subtitle) >= 8 else None
        options = looks_like_choices(ocr.read(frame[ry:ry + rh, rx:rx + rw]), rx, ry,
                                     **({"min_gap": min_gap} if min_gap else {}))

        if options:
            stamp = int(time.time())
            cv2.imwrite(str(args.out / f"{stamp}-choice.jpg"), frame)
            (args.out / f"{stamp}-choice.json").write_text(json.dumps(
                {"detected_epoch_s": stamp, "options": options,
                 "presses": presses, "subtitle": subtitle}, indent=1))
            print("CHOICE DETECTED (stopped without pressing)")
            for option in options:
                print(f"  {option['text']!r} at {option['bbox']}")
            return 0

        screen, confidence, _features, _boxes = perception.analyze(frame)
        if screen in {"dialogue", "dialogue_bark", "unknown", "transition"}:
            if not args.dry_run:
                act(base, {"type": "tap", "x": ADVANCE_POINT[0], "y": ADVANCE_POINT[1]})
            presses += 1
            if presses % 10 == 0:
                print(f"{presses} presses | {screen} ({confidence:.2f}) | last: {subtitle[:60]!r}")
            time.sleep(args.interval)
            continue

        # Anything else (reward, menu, loading) is not a conversation to press through.
        print(f"stopped: screen is {screen} ({confidence:.2f})")
        return 1

    print("time budget exhausted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
