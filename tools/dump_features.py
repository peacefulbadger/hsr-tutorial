"""Dump features for labelled frames, so thresholds can be set from evidence.

    python tools/dump_features.py data/phase1/labels.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.game_hsr.features import extract
from adapters.game_hsr.regions import FEATURE_REGIONS


def main(argv: list[str]) -> int:
    records = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    rows = []
    for record in records:
        if not record.get("label"):
            continue
        frame = cv2.imread(record["path"])
        if frame is None:
            continue
        features = extract(frame, FEATURE_REGIONS).as_dict()
        rows.append((record["label"], features, record["path"]))

    header = f"{'truth':<10}{'bright':>8}{'band_drk':>10}{'text_px':>9}{'teal':>8}{'party':>8}{'enemy_hp':>10}{'edges':>8}  path"
    print(header)
    for label, features, path in rows:
        print(
            f"{label:<10}{features['mean_brightness']:>8.1f}"
            f"{features['dialogue_band_dark']:>10.3f}"
            f"{features['dialogue_text_pixels']:>9.4f}"
            f"{features['teal_ring_pixels']:>8.3f}"
            f"{features['party_panel_bright']:>8.4f}"
            f"{features['enemy_hp_bar_pixels']:>10.4f}"
            f"{features['edge_density']:>8.3f}  {Path(path).name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
