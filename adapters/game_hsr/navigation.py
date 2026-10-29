"""Plan a route across the minimap, around the lines.

The white lines on the disc are geometry: walls, room boundaries, and the desks and
planters standing in the room. Walking at the objective marker ignores all of it, which is
why the character parks itself against a terminal desk with the objective 8 m away and the
distance never changes.

So the disc is treated as a small occupancy grid and the route is planned in it:

  obstacles   bright, desaturated pixels -- the white lines. The gold objective diamonds
              and the cyan player arrow are overlays drawn *on top* of the map, so they
              are subtracted first; leaving them in erases the walls they happen to cover.
  free space  everything else inside the disc
  start/goal  the arrow's centroid and the objective pip's centroid

A breadth-first search over 4-connected cells cannot cut a diagonal through a wall corner,
which matters here: a diagonal shortcut through a doorway frame is exactly the mistake
that wedges the character. The first waypoint of the path is then reported as a bearing in
the same north-up frame the arrow's facing is measured in, so the caller can turn toward
the waypoint instead of toward the objective.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from adapters.game_hsr import minimap

# Cells are 4 disc pixels: fine enough to thread a doorway, coarse enough that BFS over a
# 34x34 grid costs nothing.
CELL = 4
DISC_RADIUS = minimap.MINIMAP_RADIUS - 4
# A cell counts as blocked when this much of it is line, so a thin antialiased line still
# registers without the whole disc turning solid.
BLOCK_FRACTION = 0.5
# Lines are found by saturation, not brightness: see obstacle_mask. Measured on live
# discs the lines sit at S 0-28 and the floor at S 39-84.
LINE_SAT_MAX = 45
LINE_VAL_MIN = 150
# The character is not a point; inflate obstacles by this many cells so a route does not
# graze a wall and stop dead on it.
INFLATE_CELLS = 1
# How far ahead the steering aim point sits, in cells.
LOOKAHEAD_CELLS = 3


@dataclass(frozen=True)
class Route:
    """The planned route, as the caller needs it."""

    bearing_deg: float | None      # to the next waypoint, 0 = up/north, clockwise
    straight_bearing_deg: float | None  # to the objective itself, for comparison
    blocked: bool                  # no route found through the lines
    detour_deg: float | None       # how far the route bends away from the direct line
    waypoint: tuple[int, int] | None
    path: list[tuple[int, int]] = field(default_factory=list)

    @property
    def needs_detour(self) -> bool:
        """True when walking straight at the marker would hit something."""
        return self.detour_deg is not None and abs(self.detour_deg) > 20.0


def _disc(frame_bgr: np.ndarray) -> np.ndarray:
    cx, cy = minimap.MINIMAP_CENTRE
    return frame_bgr[cy - DISC_RADIUS:cy + DISC_RADIUS,
                     cx - DISC_RADIUS:cx + DISC_RADIUS]


def obstacle_mask(disc: np.ndarray) -> np.ndarray:
    """White lines, with the overlay icons taken back out.

    Brightness does not separate walls from floor: measured on a live disc, wall pixels
    have value 46-75 while the floor beside them reads 78-141, and the floor of the next
    room is darker than this room's walls. Saturation does separate them -- the lines are
    white/grey (S 0-28) and the floor is tinted blue-grey (S 39-84) -- so that is what the
    mask keys on.
    """
    hsv = cv2.cvtColor(disc, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    lines = ((sat < LINE_SAT_MAX) & (val > LINE_VAL_MIN)).astype(np.uint8)

    # The objective diamonds are gold, the player arrow cyan; both are drawn over the map
    # and both are large enough (17-30 px across) to hide the walls underneath.
    gold = ((hue >= 15) & (hue <= 42) & (sat > 130) & (val > 170)).astype(np.uint8)
    cyan = ((hue >= 88) & (hue <= 106) & (sat > 150) & (val > 170)).astype(np.uint8)
    icons = cv2.dilate(np.maximum(gold, cyan), np.ones((5, 5), np.uint8))
    lines[icons > 0] = 0

    # Outside the disc there is no floor to walk on.
    radius = min(disc.shape[:2]) // 2
    ys, xs = np.mgrid[0:disc.shape[0], 0:disc.shape[1]]
    outside = ((xs - radius) ** 2 + (ys - radius) ** 2) > (radius - 1) ** 2
    lines[outside] = 1
    return lines


def _grid(mask: np.ndarray) -> np.ndarray:
    """Blocked cells, inflated so routes keep clear of walls."""
    rows, cols = mask.shape[0] // CELL, mask.shape[1] // CELL
    blocked = np.zeros((rows, cols), bool)
    for row in range(rows):
        for col in range(cols):
            patch = mask[row * CELL:(row + 1) * CELL, col * CELL:(col + 1) * CELL]
            blocked[row, col] = patch.mean() >= BLOCK_FRACTION
    if INFLATE_CELLS:
        blocked = cv2.dilate(blocked.astype(np.uint8),
                             np.ones((2 * INFLATE_CELLS + 1,) * 2, np.uint8)) > 0
    return blocked


def _cell_of(point: tuple[float, float]) -> tuple[int, int]:
    return int(point[1] // CELL), int(point[0] // CELL)


def _clear_around(blocked: np.ndarray, cell: tuple[int, int], radius: int = 1) -> None:
    """Open a small pocket around a point that an icon was covering.

    The goal sits under a 30 px diamond, so inflation can seal it in. Clearing its own
    cell (and the neighbours) is what makes the target reachable at all.
    """
    row, col = cell
    blocked[max(0, row - radius):row + radius + 1,
            max(0, col - radius):col + radius + 1] = False


def _bfs(blocked: np.ndarray, start: tuple[int, int],
         goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    rows, cols = blocked.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return None
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return None

    queue = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            node: tuple[int, int] | None = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            return path[::-1]
        row, col = current
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (row + d_row, col + d_col)
            if not (0 <= neighbour[0] < rows and 0 <= neighbour[1] < cols):
                continue
            if blocked[neighbour] or neighbour in came_from:
                continue
            came_from[neighbour] = current
            queue.append(neighbour)
    return None


def _bearing(from_px: tuple[float, float], to_px: tuple[float, float]) -> float:
    dx = to_px[0] - from_px[0]
    dy = to_px[1] - from_px[1]
    return math.degrees(math.atan2(dx, -dy)) % 360.0  # 0 = up, clockwise


def plan(frame_bgr: np.ndarray, view: minimap.MinimapView | None = None) -> Route:
    """Route from the player to the objective, around the lines."""
    if view is None:
        view = minimap.read(frame_bgr)
    straight = view.bearing_deg
    if not view.has_pip or view.facing_deg is None:
        return Route(None, straight, False, None, None)

    disc = _disc(frame_bgr)
    blocked = _grid(obstacle_mask(disc))
    player = (view.player_x, view.player_y)
    goal = (view.pip_x, view.pip_y)

    start_cell = _cell_of(player)
    goal_cell = _cell_of(goal)
    _clear_around(blocked, start_cell, radius=1)
    _clear_around(blocked, goal_cell, radius=1)

    path = _bfs(blocked, start_cell, goal_cell)
    if path is None:
        return Route(None, straight, True, None, None)
    if len(path) < 2:
        return Route(straight, straight, False, 0.0, goal, path)

    step = min(LOOKAHEAD_CELLS, len(path) - 1)
    waypoint_cell = path[step]
    waypoint = (waypoint_cell[1] * CELL + CELL / 2.0, waypoint_cell[0] * CELL + CELL / 2.0)
    bearing = _bearing(player, waypoint)

    # How far the *route* bends, not just the next waypoint: steering aims a few cells
    # ahead, so a wall that only forces a turn five cells away leaves the first waypoint
    # dead straight. Measuring the deviation over the whole path is what tells the caller
    # "this objective is not walkable in a straight line".
    detour = None
    if straight is not None:
        deviations = [
            ((_bearing(player, (cell[1] * CELL + CELL / 2.0, cell[0] * CELL + CELL / 2.0))
              - straight + 180.0) % 360.0) - 180.0
            for cell in path[1:]
        ]
        if deviations:
            detour = max(deviations, key=abs)
    return Route(bearing, straight, False, detour, (int(waypoint[0]), int(waypoint[1])), path)
