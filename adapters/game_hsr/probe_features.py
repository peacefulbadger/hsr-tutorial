"""Print feature distributions per replay, so thresholds are chosen from data.

    python -m adapters.game_hsr.probe_features data/replays/prologue-01 data/replays/combat-01
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from adapters.game_hsr.features import extract
from adapters.game_hsr.regions import FEATURE_REGIONS


def iter_frames(replay: Path, stride: int = 1):
    entries = [
        json.loads(line)
        for line in (replay / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for entry in entries[::stride]:
        frame = cv2.imread(str(replay / entry["path"]))
        if frame is None:
            continue
        yield entry, frame


def main(argv: list[str]) -> int:
    for arg in argv:
        replay = Path(arg)
        values: dict[str, list[float]] = {}
        for _entry, frame in iter_frames(replay, stride=3):
            features = extract(frame, FEATURE_REGIONS).as_dict()
            for key, value in features.items():
                values.setdefault(key, []).append(value)
        print(f"\n{replay.name}  ({len(next(iter(values.values())))} frames sampled)")
        for key, series in values.items():
            arr = np.asarray(series)
            print(f"  {key:24s} min={arr.min():10.4f}  p10={np.percentile(arr, 10):10.4f}"
                  f"  p50={np.percentile(arr, 50):10.4f}  p90={np.percentile(arr, 90):10.4f}"
                  f"  max={arr.max():10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
