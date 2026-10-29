"""Build a contact sheet from a replay, for eyeballing what a session contains.

    python tools/make_contact.py data/replays/prologue-01 --n 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contact sheet for a replay buffer")
    parser.add_argument("replay", type=Path)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    replay = args.replay
    entries = [
        json.loads(line)
        for line in (replay / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise SystemExit("empty replay")

    n = min(args.n, len(entries))
    picks = [entries[int(i * (len(entries) - 1) / max(n - 1, 1))] for i in range(n)]

    tile_w, tile_h = 320, 180
    rows = (n + args.cols - 1) // args.cols
    sheet = np.zeros((tile_h * rows, tile_w * args.cols, 3), np.uint8)

    for k, entry in enumerate(picks):
        image = cv2.imread(str(replay / entry["path"]))
        image = cv2.resize(image, (tile_w, tile_h))
        label = f"#{k:02d} {entry['t_rel_ms'] // 1000}s"
        cv2.putText(image, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        row, col = divmod(k, args.cols)
        sheet[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = image

    out = args.out or replay / "contact.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out} ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
