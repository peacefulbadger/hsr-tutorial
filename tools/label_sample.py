"""Build a labelling sheet + manifest from one or more replays.

    python tools/label_sample.py --n 36 --out data/phase1/sample.jpg \
        --manifest data/phase1/sample.json data/replays/prologue-01 data/replays/combat-01

The sheet is a numbered grid of tiles; the manifest maps tile number to
(replay, frame path, time). Labels are written separately, by a human or an
agent looking at the sheet, into the manifest's "label" field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label sample sheet")
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    per_replay = max(1, args.n // len(args.replays))
    picks = []
    for replay in args.replays:
        entries = [
            json.loads(line)
            for line in (replay / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not entries:
            continue
        count = min(per_replay, len(entries))
        for i in range(count):
            entry = entries[int(i * (len(entries) - 1) / max(count - 1, 1))]
            picks.append((replay, entry))

    tile_w, tile_h = 320, 180
    rows = (len(picks) + args.cols - 1) // args.cols
    sheet = np.zeros((tile_h * rows, tile_w * args.cols, 3), np.uint8)
    manifest = []

    for k, (replay, entry) in enumerate(picks):
        image = cv2.imread(str(replay / entry["path"]))
        thumb = cv2.resize(image, (tile_w, tile_h))
        cv2.putText(thumb, f"{k:02d} {entry['t_rel_ms'] // 1000}s", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        row, col = divmod(k, args.cols)
        sheet[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = thumb
        manifest.append({
            "index": k,
            "replay": replay.name,
            "path": str(replay / entry["path"]),
            "t_rel_ms": entry["t_rel_ms"],
            "label": None,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    args.manifest.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({len(picks)} tiles) and {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
