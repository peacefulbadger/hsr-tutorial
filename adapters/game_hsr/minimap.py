"""Read the minimap: where the player is, which way she faces, where the objective is.

The minimap holds three independent facts and they must not be confused:

  player position   the cyan arrow's *centroid*, near the disc centre
  player facing     the cyan arrow's *pointing direction* (a chevron, so the rear has
                    a notch; the facing is away from the notch)
  objective         the gold pip, offset from the arrow. When the objective is beyond
                    minimap range the pip is clamped to the rim, so the offset is a
                    reliable *direction* and a saturated *distance*.

Measured on live frames (MuMuPlayer, 1280x720):

  - camera free-look does not rotate the minimap at all (best-match rotation 0.0 deg,
    beating +-1 deg), so the disc is not a camera-relative view: a pip offset is a
    world bearing, not a "turn the camera this much" instruction.
  - the disc's contents move when the stick moves, i.e. it tracks the character.

That distinction is what makes the naive use of the pip wrong: a world bearing cannot
be turned into a camera rotation without knowing the camera's yaw. `camera_yaw` recovers
it for free whenever both markers are visible on screen: the world pip's screen x gives
the camera-relative direction to the objective, the minimap gives the world one, and the
difference is the yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Disc geometry measured off captured frames: centre, radius, and the radius at which
# a pip is considered clamped to the rim rather than genuinely that close.
MINIMAP_CENTRE = (105, 130)
MINIMAP_RADIUS = 70
PIP_RIM_PX = 52.0

# The arrow is a small, very saturated cyan dart. Thresholds are deliberately high:
# the minimap is tinted teal in places (the camera's view cone and the corridor floor),
# and a loose threshold merges that tint into the arrow and destroys its shape.
CYAN_HUE = (88, 106)
CYAN_SAT_MIN = 160
CYAN_VAL_MIN = 180
ARROW_AREA = (35, 900)


@dataclass(frozen=True)
class MinimapView:
    """One reading of the minimap disc."""

    player_x: float          # arrow centroid, disc-local pixels
    player_y: float
    facing_deg: float | None  # 0 = up/north, clockwise; None when the shape is too round
    facing_confidence: float  # 0..1, how clearly the chevron points somewhere
    pip_x: float | None       # gold pip centroid, disc-local pixels
    pip_y: float | None
    bearing_deg: float | None  # direction from player to objective, 0 = up, clockwise
    pip_distance_px: float | None
    pip_at_rim: bool          # True: direction is good, distance is clipped

    @property
    def has_pip(self) -> bool:
        return self.bearing_deg is not None

    @property
    def turn_deg(self) -> float | None:
        """Shortest signed turn (degrees, +clockwise) to face the objective."""
        if self.bearing_deg is None or self.facing_deg is None:
            return None
        delta = (self.bearing_deg - self.facing_deg + 180.0) % 360.0 - 180.0
        return delta


def _disc(frame_bgr: np.ndarray, radius: int = MINIMAP_RADIUS) -> np.ndarray:
    cx, cy = MINIMAP_CENTRE
    return frame_bgr[cy - radius:cy + radius, cx - radius:cx + radius]


def _arrow_mask(disc: np.ndarray) -> np.ndarray | None:
    """The player arrow: largest saturated-cyan blob that is not the rim decoration."""
    hsv = cv2.cvtColor(disc, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((hue >= CYAN_HUE[0]) & (hue <= CYAN_HUE[1])
            & (sat > CYAN_SAT_MIN) & (val > CYAN_VAL_MIN)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    h, w = disc.shape[:2]
    centre = np.array([w / 2.0, h / 2.0])
    best, best_score = None, -1.0
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (ARROW_AREA[0] <= area <= ARROW_AREA[1]):
            continue
        # The arrow sits near the disc centre; rim decorations are far out.
        offset = float(np.linalg.norm(centroids[i] - centre))
        if offset > 34.0:
            continue
        score = area - 4.0 * offset
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None
    blob = (labels == best).astype(np.uint8)
    return cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _facing_by_notch(blob: np.ndarray) -> tuple[float | None, float]:
    """Facing from the bite out of the icon's tail.

    The player marker is a dart: solid nose, and a V-shaped notch cut into the tail.
    Fill the convex hull and the notch is the piece of hull the icon does not cover. The
    notch is on the tail, so the nose points from the notch through the icon's body.

    This is the primary estimator because it does not care how lopsided the two wings
    are -- the wings are lit unevenly and they drag a principal-axis fit off by tens of
    degrees, while the notch stays where the icon put it.
    """
    ys, xs = np.nonzero(blob)
    if len(xs) < 20:
        return None, 0.0

    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull_mask = np.zeros_like(blob)
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(pts), 1)

    notch = ((hull_mask == 1) & (blob == 0)).astype(np.uint8)
    if int(notch.sum()) < 4:
        return None, 0.0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(notch, 8)
    if count <= 1:
        return None, 0.0
    index = max(range(1, count), key=lambda k: int(stats[k, cv2.CC_STAT_AREA]))
    area = int(stats[index, cv2.CC_STAT_AREA])
    hull_area = int(np.count_nonzero(hull_mask))
    # A real notch is a bite, not a stray antialiased pixel at the silhouette edge.
    if area < 6 or area > hull_area * 0.5:
        return None, 0.0

    notch_cx, notch_cy = (float(v) for v in centroids[index])
    vx = float(xs.mean()) - notch_cx
    vy = float(ys.mean()) - notch_cy
    distance = math.hypot(vx, vy)
    if distance < 1.5:
        return None, 0.0

    degrees = math.degrees(math.atan2(vx, -vy)) % 360.0  # 0 = up, clockwise
    confidence = min(1.0, area / 18.0) * min(1.0, distance / 4.0)
    return degrees, confidence


def _facing_by_profile(blob: np.ndarray) -> tuple[float | None, float]:
    """Fallback: facing from pixel counts at the two ends of the principal axis.

    Used when no notch is found (a solid triangle, or a notch too small to have
    survived the mask). The nose end of a dart holds fewer pixels than the tail's pair
    of wings, and this compares those two ends.
    """
    ys, xs = np.nonzero(blob)
    if len(xs) < 20:
        return None, 0.0

    pts = np.column_stack([xs, ys]).astype(np.float64)
    centred = pts - pts.mean(axis=0)
    cov = np.cov(centred.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if axis[1] < 0 or (abs(axis[1]) < 1e-9 and axis[0] < 0):
        axis = -axis

    t = centred @ axis
    n = len(t)
    if float(t.max() - t.min()) < 6.0:
        return None, 0.0
    order = np.argsort(t)
    bins = 5
    counts = [int(((order >= i * n // bins) & (order < (i + 1) * n // bins)).sum())
              for i in range(bins)]
    head, tail = counts[0], counts[-1]
    if head == tail:
        return None, 0.0

    facing = axis if head < tail else -axis
    degrees = math.degrees(math.atan2(facing[0], -facing[1])) % 360.0
    asymmetry = abs(head - tail) / max(n / bins, 1.0)
    return degrees, min(1.0, asymmetry) * 0.5  # capped: weaker evidence than a notch


def _facing(blob: np.ndarray) -> tuple[float | None, float]:
    """Facing in degrees (0 = up/north, clockwise) and how much to trust it."""
    notch_deg, notch_conf = _facing_by_notch(blob)
    if notch_deg is not None:
        return notch_deg, notch_conf
    return _facing_by_profile(blob)


def _pip_mask(disc: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(disc, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((hue >= 15) & (hue <= 42) & (sat > 130) & (val > 170)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    # The pip is a compact diamond of a few dozen pixels; the quest-panel diamonds and
    # text antialiasing live outside the disc, but the rim highlight can creep in.
    best, best_area = None, 0
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 12 or area > 500:
            continue
        if max(w, h) > 3.2 * max(min(w, h), 1):  # elongated: a rim sliver, not a diamond
            continue
        if area > best_area:
            best, best_area = i, area
    if best is None:
        return None
    return (labels == best).astype(np.uint8)


def read(frame_bgr: np.ndarray) -> MinimapView:
    """Read arrow position, facing, and objective bearing from one frame."""
    disc = _disc(frame_bgr)
    if disc.size == 0:
        raise ValueError("frame too small for the minimap region")

    arrow = _arrow_mask(disc)
    if arrow is None:
        return MinimapView(0.0, 0.0, None, 0.0, None, None, None, None, False)

    ys, xs = np.nonzero(arrow)
    player_x, player_y = float(xs.mean()), float(ys.mean())
    facing, confidence = _facing(arrow)

    pip = _pip_mask(disc)
    if pip is None:
        return MinimapView(player_x, player_y, facing, confidence,
                           None, None, None, None, False)

    pys, pxs = np.nonzero(pip)
    pip_x, pip_y = float(pxs.mean()), float(pys.mean())
    dx, dy = pip_x - player_x, pip_y - player_y
    distance = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dx, -dy)) % 360.0
    return MinimapView(
        player_x=player_x, player_y=player_y,
        facing_deg=facing, facing_confidence=confidence,
        pip_x=pip_x, pip_y=pip_y, bearing_deg=bearing,
        pip_distance_px=distance, pip_at_rim=distance >= PIP_RIM_PX - 4.0,
    )


def screen_angle(screen_offset_px: float, fov_h_deg: float = 60.0,
                 screen_half_width: int = 640) -> float:
    """Degrees right of screen centre for a horizontal pixel offset.

    The 3D view is a projection, not a sphere, so a linear map is an approximation --
    a good one near the centre, which is where the pip is during a walk. It is stated
    once here so the caller can see the model it is trusting.
    """
    return (screen_offset_px / float(screen_half_width)) * (fov_h_deg / 2.0)


def camera_yaw(bearing_deg: float | None, screen_offset_px: float | None,
               fov_h_deg: float = 60.0, screen_half_width: int = 640) -> float | None:
    """Camera yaw in the minimap frame, from the same objective seen twice.

    `bearing_deg` is the objective's world bearing from the minimap; `screen_offset_px`
    is where the world pip sits relative to screen centre (negative = left). The screen
    offset is a camera-relative angle, so the difference is the camera's yaw.

    Positive result: the camera looks clockwise of minimap-north by that many degrees.
    """
    if bearing_deg is None or screen_offset_px is None:
        return None
    return (bearing_deg - screen_angle(screen_offset_px, fov_h_deg, screen_half_width)) % 360.0


def minimap_bearing(screen_offset_px: float, yaw_deg: float,
                    fov_h_deg: float = 60.0, screen_half_width: int = 640) -> float:
    """Inverse of `camera_yaw`: the minimap-frame bearing of a screen-x offset."""
    return (yaw_deg + screen_angle(screen_offset_px, fov_h_deg, screen_half_width)) % 360.0
