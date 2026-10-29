"""Classify an entire replay and print the timeline. A sanity check on unlabelled
data: screen types should change where the session actually changed.

    python tools/classify_replay.py data/replays/prologue-01
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.game_hsr.perception import HsrPerception, RapidOcr


def main(argv: list[str]) -> int:
    replay = Path(argv[0])
    stride = int(argv[1]) if len(argv) > 1 else 5

    entries = [
        json.loads(line)
        for line in (replay / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    perception = HsrPerception(ocr=RapidOcr())
    counts: Counter[str] = Counter()
    runs: list[tuple[str, float, float]] = []

    for entry in entries[::stride]:
        frame = cv2.imread(str(replay / entry["path"]))
        if frame is None:
            continue
        screen_type, _confidence, _features = perception.classify(frame)
        counts[screen_type] += 1
        seconds = entry["t_rel_ms"] / 1000.0
        if runs and runs[-1][0] == screen_type:
            runs[-1] = (screen_type, runs[-1][1], seconds)
        else:
            runs.append((screen_type, seconds, seconds))

    total = sum(counts.values())
    print(f"{replay.name}: {total} frames sampled (every {stride}th)")
    for name, count in counts.most_common():
        print(f"  {name:<16}{count:>5}  {count / total:>6.1%}")
    print("\ntimeline:")
    for name, start, end in runs:
        if end - start >= 2.0 or name != "unknown":
            print(f"  {start:7.1f}s - {end:7.1f}s  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
