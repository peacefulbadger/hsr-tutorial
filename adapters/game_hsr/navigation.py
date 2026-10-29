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
  unknown     the floor the arrow itself is covering -- it is drawn over the map, so the
              pixels under it are not floor, they are *unseen*
  start/goal  the arrow's centroid and the objective pip's centroid

The third state is the one that matters for the desk. The dart is ~18 px across and ~20 px
long, which at the measured 0.3 m/px hides about 5 m of floor, most of it ahead. A desk
standing in front of the character sits entirely under the nose of the arrow: the lines
under it cannot be seen, so "everything not a line is free" declares the desk walkable and
the route goes straight into it. Marking those cells unknown instead, and making unknown
ground cost more than known ground, is what lets the route step *around* the blind spot
whenever there is visible floor to go round by.

A search over 4-connected cells cannot cut a diagonal through a wall corner, which matters
here: a diagonal shortcut through a doorway frame is exactly the mistake that wedges the
character. It is a weighted search rather than a plain BFS so that known floor can be
preferred without forbidding the blind spot outright -- a corridor is unseen ahead by
definition, and refusing to enter unknown cells would freeze the character in every one.
The first waypoint of the path is then reported as a bearing in the same north-up frame
the arrow's facing is measured in, so the caller can turn toward the waypoint instead of
toward the objective.

`BumpMap` covers the residue: a wall that even the unknown cells cannot explain. When the
walker stalls pushing toward a bearing, that bearing is evidence of something solid, and
it is remembered for a few frames so the planner can route around it.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from adapters.game_hsr import minimap

# Cells are 4 disc pixels: fine enough to thread a doorway, coarse enough that BFS over a
# 34x34 grid costs nothing.
CELL = 4
# The crop has to be the *same* crop `minimap.read` measured the arrow and pip in, or the
# two agree only by accident: the old 66 px radius was four pixels short of the minimap's
# 70, which put every wall one cell off from where the player was said to be standing.
DISC_RADIUS = minimap.MINIMAP_RADIUS
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
# When the objective is off the minimap the pip is pinned to the rim: a direction, not a
# place. Any free cell within this arc of that direction counts as reaching the goal.
RIM_ARC_DEG = 60.0
# When the marked cell itself is a wall (the diamond drawn over the desk it marks), the
# nearest floor this close to it is accepted instead.
GOAL_NEAR_CELLS = 2

# Cell states. The planner needs three, not two: "nothing there", "something there", and
# "no idea, the arrow is drawn over it".
FREE, BLOCKED, UNKNOWN = 0, 1, 2

# The player dart's footprint on the disc, measured: ~20 px from nose to tail, ~18 px
# across. Split asymmetrically because the tail covers ground the character has just
# walked -- that floor is known free from having been walked, whatever the pixels say --
# while everything past the nose is hidden and unproven.
ARROW_NOSE_PX = 18.0
ARROW_TAIL_PX = 4.0
ARROW_HALF_WIDTH_PX = 9.0

# Walking into unseen floor is allowed, because a corridor is unseen ahead by definition,
# but entering it costs a few cells' worth of detour, so a route that stays on visible
# floor wins unless it is much longer. The charge is a one-off toll on the step that
# crosses into unseen space, not a per-cell penalty: inside the blind spot every cell is
# equally unproven, so there is nothing to choose between them, and charging per cell
# makes the planner orbit the dart -- two or three cells sideways to "escape" a footprint
# that is re-drawn under her the moment she moves.
UNKNOWN_ENTRY_TOLL = 4.0

# A stall bumps a wall this far in front of the character -- one stride, not the edge of
# the disc. Kept short-lived: a bump outlives its evidence in a couple of seconds, and a
# stale one walls the character in where there is nothing at all.
BUMP_AHEAD_PX = 8.0
BUMP_TTL_STEPS = 3
BUMP_MERGE_DEG = 25.0


@dataclass(frozen=True)
class Route:
    """The planned route, as the caller needs it."""

    bearing_deg: float | None      # to the next waypoint, 0 = up/north, clockwise
    straight_bearing_deg: float | None  # to the objective itself, for comparison
    blocked: bool                  # no route found through the lines
    detour_deg: float | None       # how far the route bends away from the direct line
    waypoint: tuple[int, int] | None
    path: list[tuple[int, int]] = field(default_factory=list)
    unknown_cells: int = 0         # path cells the planner could not see

    @property
    def needs_detour(self) -> bool:
        """True when walking straight at the marker would hit something."""
        return self.detour_deg is not None and abs(self.detour_deg) > 20.0


def _disc(frame_bgr: np.ndarray) -> np.ndarray:
    """The minimap disc, in exactly the crop frame `minimap.read` reports coordinates in."""
    return minimap._disc(frame_bgr, DISC_RADIUS)


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


def _clear_around(state: np.ndarray, cell: tuple[int, int], radius: int = 1) -> None:
    """Open a small pocket of known floor around a point an icon was covering.

    Both ends need it. The goal sits under a 30 px gold diamond, so inflation seals it in
    and the diamond's own footprint would otherwise read as unseen. The start sits under
    the player dart, which hides its own footprint -- but the character is standing there,
    so that cell is the one piece of floor whose freeness is not in doubt, and inflation
    must not wall her into it.
    """
    row, col = cell
    state[max(0, row - radius):row + radius + 1,
          max(0, col - radius):col + radius + 1] = FREE


@dataclass
class Bump:
    """A wall the walker has run into, in the disc's north-up polar frame."""

    bearing_deg: float
    distance_px: float
    age: int = 0


class BumpMap:
    """Walls found by walking into them, remembered for a few frames.

    The stall detector is the only sensor that sees the thing under the nose of the arrow,
    and what it reports is crude: "pushing toward this bearing did not change the distance
    to the objective". That is enough to block one small patch of floor, and it is exactly
    the patch the planner keeps routing through.

    The memory is deliberately short. The disc is re-read every frame and the character
    keeps moving, so a bump recorded two seconds ago may describe floor she has already
    walked past; a bump that is still true will simply be recorded again.
    """

    TTL_STEPS = BUMP_TTL_STEPS

    def __init__(self) -> None:
        self.bumps: list[Bump] = []

    def record(self, bearing_deg: float, distance_px: float = BUMP_AHEAD_PX) -> Bump:
        """Add a bump, strengthening an existing one on a similar bearing."""
        for bump in self.bumps:
            delta = ((bearing_deg - bump.bearing_deg + 180.0) % 360.0) - 180.0
            if abs(delta) <= BUMP_MERGE_DEG:
                # Same piece of wall hit again: average the bearings rather than stacking
                # a second wall next to the first.
                bump.bearing_deg = (bump.bearing_deg + delta / 2.0) % 360.0
                bump.distance_px = max(bump.distance_px, distance_px)
                bump.age = 0
                return bump
        bump = Bump(bearing_deg % 360.0, distance_px)
        self.bumps.append(bump)
        return bump

    def age(self) -> None:
        """One frame has passed. Expire what the planner should no longer believe."""
        for bump in self.bumps:
            bump.age += 1
        self.bumps = [bump for bump in self.bumps if bump.age <= self.TTL_STEPS]

    def clear(self) -> None:
        self.bumps.clear()

    def cells(self, from_px: tuple[float, float], shape: tuple[int, int]) -> list[tuple[int, int]]:
        """The grid cells these bumps cover, for a given player position."""
        out: list[tuple[int, int]] = []
        rows, cols = shape
        for bump in self.bumps:
            radians = math.radians(bump.bearing_deg)
            x = from_px[0] + bump.distance_px * math.sin(radians)
            y = from_px[1] - bump.distance_px * math.cos(radians)
            row = min(max(int(y // CELL), 0), rows - 1)
            col = min(max(int(x // CELL), 0), cols - 1)
            out.append((row, col))
        return out

    def __bool__(self) -> bool:
        return bool(self.bumps)


def _blind_footprint(shape: tuple[int, int], view: minimap.MinimapView) -> np.ndarray:
    """Cells the player dart is drawn over: the floor the planner cannot see.

    Derived from the view rather than from the pixels because the view already carries
    everything needed -- where the arrow's centroid is and which way it points -- and
    because a footprint that comes from the same numbers the rest of the planner uses
    cannot disagree with them.
    """
    rows, cols = shape
    footprint = np.zeros((rows, cols), bool)
    if view.facing_deg is None:
        # No readable chevron: assume the blind spot is a disc, the safe over-estimate.
        for row in range(rows):
            for col in range(cols):
                x = col * CELL + CELL / 2.0 - view.player_x
                y = row * CELL + CELL / 2.0 - view.player_y
                footprint[row, col] = math.hypot(x, y) <= ARROW_NOSE_PX / 2.0
        return footprint

    radians = math.radians(view.facing_deg)
    forward = (math.sin(radians), -math.cos(radians))
    reach = ARROW_NOSE_PX + CELL
    first_row = max(0, int((view.player_y - reach) // CELL))
    last_row = min(rows - 1, int((view.player_y + reach) // CELL))
    first_col = max(0, int((view.player_x - reach) // CELL))
    last_col = min(cols - 1, int((view.player_x + reach) // CELL))
    for row in range(first_row, last_row + 1):
        for col in range(first_col, last_col + 1):
            dx = col * CELL + CELL / 2.0 - view.player_x
            dy = row * CELL + CELL / 2.0 - view.player_y
            along = dx * forward[0] + dy * forward[1]
            across = abs(dy * forward[0] - dx * forward[1])
            if -ARROW_TAIL_PX <= along <= ARROW_NOSE_PX and across <= ARROW_HALF_WIDTH_PX:
                footprint[row, col] = True
    return footprint


def occupancy(disc: np.ndarray, view: minimap.MinimapView,
              bumps: BumpMap | None = None) -> np.ndarray:
    """The three-state grid the search runs on."""
    state = np.where(_grid(obstacle_mask(disc)), BLOCKED, FREE).astype(np.uint8)
    # A bump is positive evidence of something solid, so it wins over both free and
    # unseen -- and if the lines were visible there, bounding it changes nothing.
    if bumps:
        for row, col in bumps.cells((view.player_x, view.player_y), state.shape):
            state[row, col] = BLOCKED
    blind = _blind_footprint(state.shape, view)
    # Do not turn "this cell is a wall" into "no idea": the lines under the arrow are
    # still the best evidence there is, and the arrow only hides floor.
    state[blind & (state == FREE)] = UNKNOWN
    return state


def _step_cost(state: np.ndarray, here: tuple[int, int], there: tuple[int, int]) -> float:
    """Price of one grid step: normal inside a region, extra to cross into unseen floor.

    The toll is charged on the *crossing*, not on every unseen cell. Charging per cell
    makes the planner prefer to leave the blind spot sideways, and because the blind spot
    is redrawn under the dart every time the character moves, "sideways" stays sideways:
    on open floor the route becomes a permanent staircase instead of walking at the goal.
    """
    if state[there] == BLOCKED:
        return math.inf
    if state[there] == UNKNOWN and state[here] != UNKNOWN:
        return 1.0 + UNKNOWN_ENTRY_TOLL
    return 1.0


def _search(edge_cost, shape: tuple[int, int], start: tuple[int, int],
            goal) -> list[tuple[int, int]] | None:
    """Least-cost 4-connected path, or None when the goal is walled off.

    `edge_cost(from_cell, to_cell)` returns the price of one step, or inf for "no step".
    Taking a callable rather than a per-cell array is what lets entering unseen space be
    charged once instead of once per cell.

    `goal` is either one cell or a predicate. The predicate is what makes an objective
    beyond minimap range plannable: the pip is then clamped to the rim, so it names a
    direction, not a place, and any cell on that side of the disc will do.
    """
    rows, cols = shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return None
    if callable(goal):
        reached = goal
    else:
        if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
            return None
        reached = (lambda cell, target=tuple(goal): cell == target)

    # The counter breaks ties, so an equal-cost choice is always resolved the same way.
    # Without it a symmetric room produces a different route every frame and the stick
    # angle jitters between left and right.
    tie = itertools.count()
    best = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: list[tuple[float, int, tuple[int, int]]] = [(0.0, next(tie), start)]
    while queue:
        total, _, current = heapq.heappop(queue)
        if total > best.get(current, float("inf")):
            continue
        if reached(current):
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
            step = float(edge_cost(current, neighbour))
            if not math.isfinite(step):
                continue
            tentative = total + step
            if tentative < best.get(neighbour, float("inf")):
                best[neighbour] = tentative
                came_from[neighbour] = current
                heapq.heappush(queue, (tentative, next(tie), neighbour))
    return None


def _bearing(from_px: tuple[float, float], to_px: tuple[float, float]) -> float:
    dx = to_px[0] - from_px[0]
    dy = to_px[1] - from_px[1]
    return math.degrees(math.atan2(dx, -dy)) % 360.0  # 0 = up, clockwise


def _rim_distance(cell: tuple[int, int]) -> float:
    """How far a cell is from the disc centre, in disc pixels."""
    centre = DISC_RADIUS
    return math.hypot(cell[1] * CELL + CELL / 2.0 - centre,
                      cell[0] * CELL + CELL / 2.0 - centre)


def _off_bearing(cell: tuple[int, int], view: minimap.MinimapView) -> bool:
    """Is this cell roughly in the direction the objective lies?"""
    if view.bearing_deg is None:
        return True
    cell_px = (cell[1] * CELL + CELL / 2.0, cell[0] * CELL + CELL / 2.0)
    delta = (_bearing((view.player_x, view.player_y), cell_px) - view.bearing_deg
             + 180.0) % 360.0 - 180.0
    return abs(delta) <= RIM_ARC_DEG / 2.0


def plan(frame_bgr: np.ndarray, view: minimap.MinimapView | None = None,
         bumps: BumpMap | None = None) -> Route:
    """Route from the player to the objective, around the lines and around the blind spot."""
    if view is None:
        view = minimap.read(frame_bgr)
    straight = view.bearing_deg
    if not view.has_pip or view.facing_deg is None:
        return Route(None, straight, False, None, None)

    disc = _disc(frame_bgr)
    state = occupancy(disc, view, bumps)
    player = (view.player_x, view.player_y)
    goal = (view.pip_x, view.pip_y)

    start_cell = _cell_of(player)
    goal_cell = _cell_of(goal)
    _clear_around(state, start_cell, radius=1)

    search = lambda target: _search(  # noqa: E731 - one-line adapter, used three times
        lambda here, there: _step_cost(state, here, there), state.shape, start_cell, target)

    if view.pip_at_rim:
        # Beyond minimap range the pip is pinned to the rim, so it names a direction and
        # not a place. Planning to that exact rim cell asks "get to that one wall tile",
        # which is how routes end up setting off the wrong way round the room and bending
        # 178 deg to come back. Ask instead "leave the disc heading at the objective".
        def on_the_way_out(cell):
            return _rim_distance(cell) >= DISC_RADIUS - 3 * CELL and _off_bearing(cell, view)

        path = search(on_the_way_out)
        if path is None:
            # Nothing reachable at the rim: fall back to the direct line, which at least
            # allows a route part of the way and reports the bend honestly.
            _clear_around(state, goal_cell, radius=1)
            path = search(goal_cell)
    else:
        _clear_around(state, goal_cell, radius=1)
        path = search(goal_cell)
        if path is None:
            # The pip can sit on top of the desk it marks, and then its cell is a wall.
            # Accept the nearest floor to it rather than declaring the objective sealed.
            _clear_around(state, goal_cell, radius=1)
            path = search(lambda cell: (abs(cell[0] - goal_cell[0])
                                        + abs(cell[1] - goal_cell[1])) <= GOAL_NEAR_CELLS)

    if path is None:
        return Route(None, straight, True, None, None)
    if len(path) < 2:
        return Route(straight, straight, False, 0.0, goal, path)

    # How much of the route rests on floor nobody can see. The first few cells are always
    # unseen -- they are under the arrow -- so this is reported rather than acted on; it is
    # the number to look at when a route is doing something hard to explain.
    hidden = sum(1 for cell in path[1:] if state[cell] == UNKNOWN)

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
    return Route(bearing, straight, False, detour, (int(waypoint[0]), int(waypoint[1])),
                 path, unknown_cells=hidden)
