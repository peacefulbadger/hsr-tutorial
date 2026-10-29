"""Report what the minimap says: player position, facing, objective bearing.

A diagnostic for one question the rest of the code depends on -- which way is the
character actually facing, and where is the objective relative to that -- so it prints
the same fields the walker's control law consumes, and can draw them back onto the disc
for a visual check.

    python tools/minimap_probe.py
    python tools/minimap_probe.py --annotate          # save disc-turns-*.png
    python tools/minimap_probe.py --turn 320          # rotate the camera first
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.game_hsr import minimap  # noqa: E402

SCALE = 5
CROP_RADIUS = 62


def fetch(base: str) -> np.ndarray:
    with urllib.request.urlopen(f"{base}/frame", timeout=10) as response:
        return cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)


def act(base: str, body: dict) -> None:
    request = urllib.request.Request(
        f"{base}/action", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    urllib.request.urlopen(request, timeout=15).read()


def annotate(frame: np.ndarray, view: minimap.MinimapView) -> np.ndarray:
    cx, cy = minimap.MINIMAP_CENTRE
    crop = frame[cy - CROP_RADIUS:cy + CROP_RADIUS, cx - CROP_RADIUS:cx + CROP_RADIUS]
    panel = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    px, py = int(view.player_x * SCALE), int(view.player_y * SCALE)

    if view.facing_deg is not None:
        radians = math.radians(view.facing_deg)
        tip = (int(px + 45 * math.sin(radians)), int(py - 45 * math.cos(radians)))
        cv2.arrowedLine(panel, (px, py), tip, (0, 0, 255), 3, tipLength=0.3)
        cv2.putText(panel, f"facing {view.facing_deg:5.1f} (c={view.facing_confidence:.2f})",
                    (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(panel, "facing unknown", (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    if view.has_pip and view.turn_deg is not None:
        radians = math.radians(view.bearing_deg)
        ex = int(px + view.pip_distance_px * SCALE * math.sin(radians))
        ey = int(py - view.pip_distance_px * SCALE * math.cos(radians))
        cv2.line(panel, (px, py), (ex, ey), (0, 255, 0), 2)
        cv2.putText(panel, f"objective {view.bearing_deg:5.1f} rim={view.pip_at_rim}",
                    (6, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(panel, f"turn {view.turn_deg:+6.1f} deg", (6, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return panel


def report(view: minimap.MinimapView) -> str:
    return (
        f"player=({view.player_x:5.1f},{view.player_y:5.1f}) "
        f"facing={view.facing_deg if view.facing_deg is None else round(view.facing_deg, 1)}"
        f" c={view.facing_confidence:.2f} "
        f"bearing={view.bearing_deg if view.bearing_deg is None else round(view.bearing_deg, 1)} "
        f"dist={None if view.pip_distance_px is None else round(view.pip_distance_px, 1)} "
        f"rim={view.pip_at_rim} "
        f"turn={view.turn_deg if view.turn_deg is None else round(view.turn_deg, 1)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the minimap")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--turn", type=int, default=0,
                        help="rotate the camera by this many px first (0 = no turn)")
    parser.add_argument("--annotate", action="store_true", help="save annotated crops")
    parser.add_argument("--out", type=Path, default=Path("data/phase1"))
    args = parser.parse_args(argv)

    base = args.url.rstrip("/")
    frame = fetch(base)
    readings = [("before", frame)]

    if args.turn:
        act(base, {"type": "swipe", "x": 950, "y": 360,
                   "x2": 950 + args.turn, "y2": 360, "duration_ms": 650})
        time.sleep(1.2)
        readings.append(("after", fetch(base)))

    if args.annotate:
        args.out.mkdir(parents=True, exist_ok=True)

    panels = []
    for label, image in readings:
        view = minimap.read(image)
        print(f"{label:>6}: {report(view)}")
        if args.annotate:
            panel = annotate(image, view)
            cv2.putText(panel, label, (6, panel.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            panels.append(panel)

    if panels:
        cv2.imwrite(str(args.out / "minimap-probe.png"), np.hstack(panels))
        print("wrote", args.out / "minimap-probe.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
