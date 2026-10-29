"""Tests for the minimap route planner.

Routing is tested against the occupancy grid directly, because that is where the decision
lives: given a wall, does the route bend around it? Synthesising whole frames to test that
keeps tripping over how the icons are drawn, which is a different question -- so the image
side is covered by one focused test of the mask, and the rest are grids.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from adapters.game_hsr import minimap, navigation

MID = navigation.DISC_RADIUS
ARROW_SHAPE = [(0, -8), (7, 7), (0, 3), (-7, 7)]


def signed(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def blank_map() -> np.ndarray:
    frame = np.zeros((720, 1280, 3), np.uint8)
    cx, cy = minimap.MINIMAP_CENTRE
    cv2.circle(frame, (cx, cy), minimap.MINIMAP_RADIUS - 1, (60, 60, 60), -1)
    return frame


def stub_view(*, player=(MID, MID + 14), pip=(MID, MID - 40), facing=0.0):
    player_x, player_y = player
    pip_x, pip_y = pip
    bearing = (np.degrees(np.arctan2(pip_x - player_x, -(pip_y - player_y)))) % 360.0
    return minimap.MinimapView(
        player_x=player_x, player_y=player_y, facing_deg=facing, facing_confidence=1.0,
        pip_x=pip_x, pip_y=pip_y, bearing_deg=bearing,
        pip_distance_px=float(np.hypot(pip_x - player_x, pip_y - player_y)),
        pip_at_rim=False,
    )


def mask_with_wall(gap_on_the_left: bool = True) -> np.ndarray:
    """A wall across the disc at row 46-50, with a gap on one side."""
    mask = np.zeros((2 * MID, 2 * MID), np.uint8)
    if gap_on_the_left:
        mask[46:50, 44:2 * MID] = 1
    else:
        mask[46:50, 0:2 * MID - 44] = 1
    return mask


def path_bearings(route) -> list[float]:
    """Signed bearing from the player to every point on the route."""
    start = route.path[0]
    start_px = (start[1] * navigation.CELL + navigation.CELL / 2.0,
                start[0] * navigation.CELL + navigation.CELL / 2.0)
    return [signed(navigation._bearing(
        start_px, (cell[1] * navigation.CELL + navigation.CELL / 2.0,
                   cell[0] * navigation.CELL + navigation.CELL / 2.0)))
        for cell in route.path[1:]]


def test_route_bends_toward_the_gap_in_a_wall(monkeypatch):
    view = stub_view()
    monkeypatch.setattr(navigation, "obstacle_mask", lambda disc: mask_with_wall())
    route = navigation.plan(np.zeros((720, 1280, 3), np.uint8), view)
    assert route.bearing_deg is not None
    assert route.needs_detour, f"expected a detour, got {route.detour_deg}"
    # The route reaches left of straight up somewhere, because that is where the gap is.
    assert min(path_bearings(route)) < -15


def test_route_bends_the_other_way_when_the_gap_moves(monkeypatch):
    view = stub_view()
    monkeypatch.setattr(navigation, "obstacle_mask", lambda disc: mask_with_wall(False))
    route = navigation.plan(np.zeros((720, 1280, 3), np.uint8), view)
    assert route.bearing_deg is not None
    assert route.needs_detour, f"expected a detour, got {route.detour_deg}"
    assert max(path_bearings(route)) > 15


def test_open_floor_gives_the_straight_bearing(monkeypatch):
    view = stub_view()
    empty = np.zeros((2 * MID, 2 * MID), np.uint8)
    monkeypatch.setattr(navigation, "obstacle_mask", lambda disc: empty)
    route = navigation.plan(np.zeros((720, 1280, 3), np.uint8), view)
    assert route.bearing_deg is not None
    assert abs(signed(route.bearing_deg)) < 20
    assert not route.needs_detour


def test_a_sealed_room_reports_blocked_instead_of_going_straight(monkeypatch):
    view = stub_view()
    mask = np.zeros((2 * MID, 2 * MID), np.uint8)
    # A closed ring around the objective.
    mask[16:20, 40:100] = 1
    mask[60:64, 40:100] = 1
    mask[16:64, 40:44] = 1
    mask[16:64, 96:100] = 1
    monkeypatch.setattr(navigation, "obstacle_mask", lambda disc: mask)
    route = navigation.plan(np.zeros((720, 1280, 3), np.uint8), view)
    assert route.blocked is True
    assert route.bearing_deg is None


def test_no_pip_means_no_route():
    view = minimap.MinimapView(player_x=MID, player_y=MID, facing_deg=0.0,
                               facing_confidence=1.0, pip_x=None, pip_y=None,
                               bearing_deg=None, pip_distance_px=None, pip_at_rim=False)
    route = navigation.plan(np.zeros((720, 1280, 3), np.uint8), view)
    assert route.bearing_deg is None
    assert route.straight_bearing_deg is None


def test_obstacle_mask_finds_lines_and_ignores_icons():
    cx, cy = minimap.MINIMAP_CENTRE
    radius = navigation.DISC_RADIUS
    ys, xs = np.mgrid[0:2 * radius, 0:2 * radius]
    inside = ((xs - radius) ** 2 + (ys - radius) ** 2) < (radius - 6) ** 2

    frame = blank_map()
    cv2.line(frame, (cx - 40, cy - 20), (cx + 30, cy - 20), (255, 255, 255), 4)
    with_line = navigation.obstacle_mask(navigation._disc(frame))
    line_pixels = int((with_line.astype(bool) & inside).sum())
    assert line_pixels > 50, "the wall should register as an obstacle"

    icons_only = blank_map()
    hsv = cv2.cvtColor(icons_only, cv2.COLOR_BGR2HSV)
    icon = np.array([[cx, cy - 30], [cx + 16, cy - 12], [cx, cy + 6], [cx - 16, cy - 12]],
                    np.int32)
    cv2.polylines(hsv, [icon], True, (25, 200, 255), 3)
    without = navigation.obstacle_mask(navigation._disc(
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)))
    icon_pixels = int((without.astype(bool) & inside).sum())
    assert icon_pixels < line_pixels / 2, "the objective icon must not read as geometry"


def test_bearing_zero_is_up_and_positive_is_clockwise():
    assert navigation._bearing((50, 100), (50, 40)) == pytest.approx(0.0, abs=1)
    assert navigation._bearing((50, 100), (110, 100)) == pytest.approx(90.0, abs=1)
    assert navigation._bearing((50, 100), (50, 160)) == pytest.approx(180.0, abs=1)
    assert navigation._bearing((50, 100), (-10, 100)) == pytest.approx(270.0, abs=1)


# --- A: the floor the player dart is drawn over is unseen, not free -------------------

GRID = (2 * MID // navigation.CELL, 2 * MID // navigation.CELL)


def open_floor(monkeypatch) -> np.ndarray:
    empty = np.zeros((2 * MID, 2 * MID), np.uint8)
    monkeypatch.setattr(navigation, "obstacle_mask", lambda disc: empty)
    return empty


def state_grid(monkeypatch, view) -> np.ndarray:
    open_floor(monkeypatch)
    return navigation.occupancy(np.zeros((2 * MID, 2 * MID, 3), np.uint8), view)


def test_the_arrow_hides_the_floor_ahead_and_spares_the_floor_behind():
    view = stub_view(facing=0.0)
    footprint = navigation._blind_footprint(GRID, view)
    player = navigation._cell_of((view.player_x, view.player_y))
    row, col = player

    assert footprint[row - 3, col], "the floor under the nose of the dart is unseen"
    assert footprint[row, col], "the cell the character stands on is covered by the dart"
    assert not footprint[row + 3, col], "the floor she has just walked is not blind"
    assert not footprint[row, col + 3], "the floor beside her is visible"


def test_the_arrow_turns_the_blind_spot_with_it():
    right = navigation._blind_footprint(GRID, stub_view(facing=90.0))
    player = navigation._cell_of((MID, MID + 14))
    row, col = player
    assert right[row, col + 3], "facing right: the unseen floor is to the right"
    assert not right[row - 3, col], "facing right: straight ahead is visible"


def test_a_wall_under_the_arrow_is_still_a_wall(monkeypatch):
    # "Unknown" must not overwrite the one thing the disc does say: those pixels were
    # tested for line-ness before the arrow's footprint was even considered.
    view = stub_view(facing=0.0)
    row, col = navigation._cell_of((view.player_x, view.player_y))
    wall = np.zeros((2 * MID, 2 * MID), np.uint8)
    wall[row * navigation.CELL:(row + 1) * navigation.CELL,
         col * navigation.CELL:(col + 1) * navigation.CELL] = 1
    monkeypatch.setattr(navigation, "obstacle_mask", lambda disc: wall)
    state = navigation.occupancy(np.zeros((2 * MID, 2 * MID, 3), np.uint8), view)
    assert state[row, col] == navigation.BLOCKED


def test_crossing_into_unseen_floor_is_charged_once(monkeypatch):
    # The invariant that stops the route orbiting the dart: a second step deeper into the
    # blind spot is free, so the planner has no reason to prefer one unseen cell over
    # another and will not sidestep to "escape" it.
    view = stub_view(facing=0.0)
    state = state_grid(monkeypatch, view)
    row, col = navigation._cell_of((view.player_x, view.player_y))
    free_cell = (row + 5, col + 6)
    unseen_cell = (row - 3, col)
    assert state[free_cell] == navigation.FREE
    assert state[unseen_cell] == navigation.UNKNOWN

    assert navigation._step_cost(state, free_cell, unseen_cell) > 2.0
    assert navigation._step_cost(state, unseen_cell, unseen_cell) == 1.0
    assert navigation._step_cost(state, unseen_cell, free_cell) == 1.0
    state[row - 4, col] = navigation.BLOCKED
    assert navigation._step_cost(state, unseen_cell, (row - 4, col)) == float("inf")


def test_open_floor_still_walks_at_the_objective(monkeypatch):
    # The regression that A must not introduce: with nothing visible in the way, the
    # route has to aim at the goal instead of stepping around its own footprint.
    view = stub_view()
    open_floor(monkeypatch)
    route = navigation.plan(np.zeros((720, 1280, 3), np.uint8), view)
    assert route.bearing_deg is not None
    assert abs(signed(route.bearing_deg)) < 20, f"route orbited: {route.bearing_deg}"
    assert not route.needs_detour
    assert route.unknown_cells > 0, "the route does cross the dart's footprint"


def test_walking_out_of_the_blind_spot_sees_every_cell(monkeypatch):
    # Straight backwards is the one direction the dart does not cover (its tail is 4 px),
    # so the count has to be able to read zero -- otherwise it is a constant, not a
    # measurement, and `unknown_cells` would be reporting nothing at all.
    view = stub_view(player=(MID, MID + 14), pip=(MID, MID + 40), facing=0.0)
    open_floor(monkeypatch)
    route = navigation.plan(np.zeros((720, 1280, 3), np.uint8), view)
    assert route.bearing_deg is not None
    assert route.unknown_cells == 0


# --- B: walls found by walking into them ---------------------------------------------


def test_bump_map_expires_after_a_few_frames():
    bumps = navigation.BumpMap()
    bumps.record(0.0)
    assert bumps
    for _ in range(navigation.BUMP_TTL_STEPS):
        bumps.age()
    assert bumps, "a fresh bump must survive the first frames"
    bumps.age()
    assert not bumps


def test_bump_map_merges_repeated_hits_on_the_same_bearing():
    bumps = navigation.BumpMap()
    bumps.record(10.0)
    bumps.record(20.0)
    assert len(bumps.bumps) == 1, "the same wall hit twice is one wall"
    assert bumps.bumps[0].bearing_deg == pytest.approx(15.0, abs=1)

    bumps.record(90.0)
    assert len(bumps.bumps) == 2, "a different wall is a different bump"


def test_a_bump_lands_in_front_of_the_player(monkeypatch):
    view = stub_view(facing=0.0)
    open_floor(monkeypatch)
    bumps = navigation.BumpMap()
    bumps.record(0.0)
    row, col = navigation._cell_of((view.player_x, view.player_y))
    assert bumps.cells((view.player_x, view.player_y), GRID) == [(row - 2, col)]

    state = navigation.occupancy(np.zeros((2 * MID, 2 * MID, 3), np.uint8), view, bumps)
    assert state[row - 2, col] == navigation.BLOCKED


def test_a_bump_turns_a_straight_route(monkeypatch):
    # The stall only says "something is there"; turning it into a blocked cell is what
    # lets the planner go round the desk instead of re-planning the same collision.
    view = stub_view(facing=0.0)
    open_floor(monkeypatch)
    frame = np.zeros((720, 1280, 3), np.uint8)
    straight = navigation.plan(frame, view)
    assert not straight.needs_detour

    bumps = navigation.BumpMap()
    bumps.record(view.facing_deg)
    bumped = navigation.plan(frame, view, bumps)
    assert bumped.bearing_deg is not None
    assert bumped.needs_detour, f"bump ignored, detour {bumped.detour_deg}"

