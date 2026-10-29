"""Draw what the route planner sees: obstacles, player, objective, and the path.

Debugging a planner by reading its bearings is hopeless -- "detour +175 deg" could be a
correct reading of a strange room or a broken mask. This renders the disc at 8x with the
obstacle grid shaded, the path drawn, and the numbers written on it.

    python tools/plan_preview.py
    python tools/plan_preview.py --out data/phase1/plan.png
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.game_hsr import minimap, navigation  # noqa: E402

SCALE = 8


def fetch(base: str) -> np.ndarray:
    with urllib.request.urlopen(f"{base}/frame", timeout=10) as response:
        return cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)


def render(frame: np.ndarray) -> tuple[np.ndarray, str]:
    view = minimap.read(frame)
    route = navigation.plan(frame, view)
    disc = navigation._disc(frame)
    mask = navigation.obstacle_mask(disc)
    state = navigation.occupancy(disc, view)

    panel = cv2.resize(disc, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)

    # Shade the grid, so it is obvious what the planner believes: red is a wall, amber is
    # floor it cannot see because the dart is drawn over it, and unshaded is free.
    shades = {navigation.BLOCKED: (0, 0, 140), navigation.UNKNOWN: (0, 170, 255)}
    for row in range(state.shape[0]):
        for col in range(state.shape[1]):
            colour = shades.get(int(state[row, col]))
            if colour is None:
                continue
            top_left = (col * navigation.CELL * SCALE, row * navigation.CELL * SCALE)
            bottom_right = (top_left[0] + navigation.CELL * SCALE,
                            top_left[1] + navigation.CELL * SCALE)
            overlay = panel.copy()
            cv2.rectangle(overlay, top_left, bottom_right, colour, -1)
            cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, panel)

    if route.path:
        points = [(int((cell[1] * navigation.CELL + navigation.CELL / 2) * SCALE),
                   int((cell[0] * navigation.CELL + navigation.CELL / 2) * SCALE))
                  for cell in route.path]
        for first, second in zip(points, points[1:]):
            cv2.line(panel, first, second, (0, 255, 0), 2)
        cv2.circle(panel, points[-1], 5, (0, 255, 0), -1)

    if view.facing_deg is not None:
        px, py = int(view.player_x * SCALE), int(view.player_y * SCALE)
        cv2.circle(panel, (px, py), 6, (255, 0, 255), -1)

    summary = (
        f"facing={view.facing_deg if view.facing_deg is None else round(view.facing_deg)} "
        f"bearing={view.bearing_deg if view.bearing_deg is None else round(view.bearing_deg)} "
        f"dist={None if view.pip_distance_px is None else round(view.pip_distance_px)} "
        f"route={route.bearing_deg if route.bearing_deg is None else round(route.bearing_deg)} "
        f"detour={route.detour_deg if route.detour_deg is None else round(route.detour_deg)} "
        f"blocked={route.blocked} cells={len(route.path)} unseen={route.unknown_cells}"
    )
    bars = [
        f"obstacle px={int(mask.sum())} walls={int((state == navigation.BLOCKED).sum())} "
        f"unseen={int((state == navigation.UNKNOWN).sum())}",
        summary,
    ]
    for index, text in enumerate(bars):
        cv2.putText(panel, text, (4, 18 + index * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
    return panel, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the planner's view")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", type=Path, default=Path("data/phase1/plan-preview.png"))
    args = parser.parse_args(argv)

    frame = fetch(args.url.rstrip("/"))
    panel, summary = render(frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), panel)
    print(summary)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
