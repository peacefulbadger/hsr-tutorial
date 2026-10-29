"""Tests for the minimap reader.

The frames are synthetic and deliberately small on detail: a 17 px cyan dart on a dark
disc with a gold diamond pip. That is the size the real game draws, and it is the size at
which a naive "farthest point from the centroid" estimator fails, so the tests pin the
behaviour that actually survives that resolution.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from adapters.game_hsr import minimap

FRAME_W, FRAME_H = 1280, 720
DART = [(0.0, -9.0), (8.0, 8.0), (0.0, 4.0), (-8.0, 8.0)]  # nose, right wing, notch, left


def _rotate(points, angle_deg):
    radians = math.radians(angle_deg)
    cos, sin = math.cos(radians), math.sin(radians)
    return [(x * cos - y * sin, x * sin + y * cos) for x, y in points]


def _frame(*, facing_deg=None, bearing_deg=None, distance=40.0,
           arrow_offset=(0.0, 0.0), clutter=False):
    hsv = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    cx, cy = minimap.MINIMAP_CENTRE
    cv2.circle(hsv, (cx, cy), minimap.MINIMAP_RADIUS, (100, 20, 30), -1)  # dark disc

    if clutter:
        # A pale teal wedge, like the camera's view cone: same family as the arrow but
        # far less saturated. It must not be mistaken for the player marker.
        wedge = np.array([[cx, cy], [cx - 60, cy - 40], [cx - 20, cy + 55]], np.int32)
        cv2.fillPoly(hsv, [wedge], (95, 70, 120))

    if facing_deg is not None:
        pts = _rotate(DART, facing_deg)
        poly = np.array([[cx + arrow_offset[0] + x, cy + arrow_offset[1] + y]
                         for x, y in pts], np.int32)
        cv2.fillPoly(hsv, [poly], (95, 220, 255))  # h, s, v: saturated cyan

    if bearing_deg is not None:
        px = cx + distance * math.sin(math.radians(bearing_deg))
        py = cy - distance * math.cos(math.radians(bearing_deg))
        diamond = np.array([[px, py - 6], [px + 5, py], [px, py + 6], [px - 5, py]], np.int32)
        cv2.fillPoly(hsv, [diamond], (25, 200, 255))  # gold

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270, 315])
def test_facing_is_recovered_from_the_notch(angle):
    view = minimap.read(_frame(facing_deg=angle))
    assert view.facing_deg is not None
    assert view.facing_confidence > 0.3
    assert abs(minimap_signed(view.facing_deg - angle)) < 8.0


@pytest.mark.parametrize("angle", [0, 40, 95, 200, 300])
def test_bearing_is_recovered_from_the_pip(angle):
    view = minimap.read(_frame(facing_deg=0, bearing_deg=angle, distance=35))
    assert view.has_pip
    assert abs(minimap_signed(view.bearing_deg - angle)) < 6.0


def minimap_signed(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


def test_turn_is_the_short_way_round():
    view = minimap.read(_frame(facing_deg=350, bearing_deg=10, distance=40))
    assert view.turn_deg == pytest.approx(20.0, abs=6.0)


def test_turn_is_negative_when_the_objective_is_behind_left():
    view = minimap.read(_frame(facing_deg=10, bearing_deg=350, distance=40))
    assert view.turn_deg == pytest.approx(-20.0, abs=6.0)


def test_pip_on_the_rim_is_flagged_as_a_clipped_distance():
    rim = minimap.read(_frame(facing_deg=0, bearing_deg=0, distance=60))
    near = minimap.read(_frame(facing_deg=0, bearing_deg=0, distance=25))
    assert rim.pip_at_rim is True
    assert near.pip_at_rim is False
    assert rim.pip_distance_px > near.pip_distance_px


def test_no_arrow_means_no_facing_no_turn_and_no_bearing():
    # The pip offset is only a bearing *relative to the player*. Without the arrow there
    # is no origin, so reporting a bearing would be inventing one.
    view = minimap.read(_frame(bearing_deg=90, distance=40))
    assert view.facing_deg is None
    assert view.turn_deg is None
    assert view.has_pip is False


def test_no_pip_means_no_bearing():
    view = minimap.read(_frame(facing_deg=90))
    assert view.facing_deg is not None
    assert view.has_pip is False
    assert view.turn_deg is None


def test_low_saturation_teal_clutter_does_not_steal_the_arrow():
    view = minimap.read(_frame(facing_deg=120, bearing_deg=300, distance=55, clutter=True))
    assert view.facing_deg is not None
    assert abs(minimap_signed(view.facing_deg - 120)) < 10.0
    assert abs(minimap_signed(view.bearing_deg - 300)) < 8.0


def test_screen_angle_is_linear_and_signed():
    assert minimap.screen_angle(0) == pytest.approx(0.0)
    assert minimap.screen_angle(320) == pytest.approx(15.0)   # half width at 60 deg fov
    assert minimap.screen_angle(-320) == pytest.approx(-15.0)


def test_camera_yaw_and_its_inverse_agree():
    for yaw in (0.0, 40.0, 300.0):
        for offset in (-300.0, 0.0, 250.0):
            bearing = minimap.minimap_bearing(offset, yaw)
            assert minimap.camera_yaw(bearing, offset) == pytest.approx(yaw, abs=1e-6)


def test_camera_yaw_needs_both_markers():
    assert minimap.camera_yaw(None, 100.0) is None
    assert minimap.camera_yaw(90.0, None) is None


# --- more than one objective diamond on the disc -------------------------------------


def _pips_frame(diamonds, facing=0.0):
    """`diamonds` is a list of (bearing_deg, distance_px, half_width_px)."""
    hsv = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    cx, cy = minimap.MINIMAP_CENTRE
    cv2.circle(hsv, (cx, cy), minimap.MINIMAP_RADIUS, (100, 20, 30), -1)
    pts = _rotate(DART, facing)
    poly = np.array([[cx + x, cy + y] for x, y in pts], np.int32)
    cv2.fillPoly(hsv, [poly], (95, 220, 255))
    for bearing, distance, half in diamonds:
        px = cx + distance * math.sin(math.radians(bearing))
        py = cy - distance * math.cos(math.radians(bearing))
        diamond = np.array([[px, py - half], [px + half, py], [px, py + half],
                            [px - half, py]], np.int32)
        cv2.fillPoly(hsv, [diamond], (25, 200, 255))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_every_diamond_on_the_disc_is_counted():
    # "Investigating the terminals 0/2" puts a labelled diamond on each terminal, and a
    # chest shares the colour. The reader has to admit there is a choice to make.
    view = minimap.read(_pips_frame([(90, 40, 5), (270, 40, 6)]))
    assert view.pip_candidates == 2
    assert view.has_pip


def test_the_largest_diamond_wins_when_there_is_nothing_to_go_on():
    view = minimap.read(_pips_frame([(90, 40, 6), (270, 40, 8)]))
    assert abs(minimap_signed(view.bearing_deg - 270.0)) < 10.0


def test_a_known_bearing_keeps_the_same_diamond_next_frame():
    # The pip must not flip between the two terminals because one sparkled brighter: a
    # flip is a 200 deg change of objective and the walker turns round and walks back.
    first = minimap.read(_pips_frame([(90, 40, 6), (270, 40, 8)]), prefer_bearing=90.0)
    second = minimap.read(_pips_frame([(90, 40, 8), (270, 40, 6)]), prefer_bearing=90.0)
    assert abs(minimap_signed(first.bearing_deg - 90.0)) < 10.0
    assert abs(minimap_signed(second.bearing_deg - 90.0)) < 10.0


def test_the_objective_is_followed_when_the_old_one_disappears():
    # Completing a terminal removes its marker, so the only diamond left is the next one.
    view = minimap.read(_pips_frame([(270, 40, 8)]), prefer_bearing=90.0)
    assert abs(minimap_signed(view.bearing_deg - 270.0)) < 10.0


def test_a_sparkle_fragment_does_not_steal_the_pip():
    # A stray antialiased blob next to a diamond can sit at almost any bearing; it is too
    # small to be a marker and must not win on the tie-break.
    view = minimap.read(_pips_frame([(90, 40, 7), (10, 30, 3)]), prefer_bearing=10.0)
    assert abs(minimap_signed(view.bearing_deg - 90.0)) < 12.0
