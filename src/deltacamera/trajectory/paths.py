"""Position path functions for camera trajectories.

All functions take an interpolation parameter t in [0, 1] and return a
3D position. Easing should be applied to t by the caller before passing
it here.
"""

import numpy as np


def lerp(a, b, t):
    """Linear interpolation between two 3D points."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return (a * (1 - t) + b * t).astype(np.float32)


def _horizontal_basis(up):
    """Unit up vector and two unit vectors spanning the plane perpendicular to it.

    The first horizontal axis is the world x axis projected into that plane
    (the world y axis if up is parallel to x), so for up = +z the basis is
    exactly (x, y, z) and azimuths are measured from +x toward +y.
    """
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - up * (ref @ up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return up, e1, e2


def cylindrical_arc(pos_a, pos_b, t, pivot, up, az_start=None):
    """Interpolate position in cylindrical coordinates around a pivot.

    Simultaneously interpolates azimuth, elevation, and radius. Traces a
    spiral when the endpoints differ in azimuth (e.g., descent from
    overhead to an orbit position at a different angle). Azimuth wraps
    via the shortest arc. Returns exact endpoints at t=0 and t=1
    (no float round-trip error).

    A start point on the pivot's vertical axis (e.g. the overhead point)
    has no azimuth of its own; by default the target's azimuth is used,
    so the path leaves the axis straight toward the target. ``az_start``
    (radians) sets the azimuth at which the path leaves the axis instead,
    swinging to the target's azimuth along the way — e.g. the azimuth of
    the preceding ascent, so a descent continues the ascent's motion and
    the upright orientation stays continuous through the pole.

    Azimuth and elevation are taken relative to ``up``: the horizontal plane
    is perpendicular to it, and azimuth is measured from the world x axis
    projected into that plane (see ``_horizontal_basis``).
    """
    if t <= 0:
        return np.asarray(pos_a, np.float32).copy()
    if t >= 1:
        return np.asarray(pos_b, np.float32).copy()
    up, e1, e2 = _horizontal_basis(up)
    pivot = np.asarray(pivot, np.float64)
    da = np.asarray(pos_a, np.float64) - pivot
    db = np.asarray(pos_b, np.float64) - pivot
    # Coordinates in the (e1, e2, up) frame
    xa, ya, za = da @ e1, da @ e2, da @ up
    xb, yb, zb = db @ e1, db @ e2, db @ up
    na, nb = np.hypot(xa, ya), np.hypot(xb, yb)
    if na > 1e-3:
        az_a = np.arctan2(ya, xa)
    elif az_start is not None:
        az_a = float(az_start)
    else:
        az_a = np.arctan2(yb, xb)
    az_b = np.arctan2(yb, xb) if nb > 1e-3 else az_a
    daz = az_b - az_a
    if daz > np.pi:
        daz -= 2 * np.pi
    if daz < -np.pi:
        daz += 2 * np.pi
    elev_a = np.arctan2(za, max(na, 1e-3))
    elev_b = np.arctan2(zb, max(nb, 1e-3))
    rad_a = np.linalg.norm(da)
    rad_b = np.linalg.norm(db)
    az = az_a + daz * t
    radius = rad_a * (1 - t) + rad_b * t
    elev = elev_a * (1 - t) + elev_b * t
    horiz = e1 * np.cos(az) + e2 * np.sin(az)
    return (pivot + horiz * (radius * np.cos(elev))
            + up * (radius * np.sin(elev))).astype(np.float32)


def circle_positions(center, radius, up, n, az_start_deg=0.0):
    """Generate n evenly spaced positions on a circle perpendicular to ``up``.

    Azimuths follow the same convention as :func:`cylindrical_arc`.

    Returns:
        ndarray shape (n, 3), float32.
    """
    center = np.asarray(center, np.float64)
    _, e1, e2 = _horizontal_basis(up)
    angles = np.radians(az_start_deg) + np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = center + radius * (np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2)
    return pts.astype(np.float32)
