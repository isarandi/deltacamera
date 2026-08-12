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


def cylindrical_arc(pos_a, pos_b, t, pivot, up):
    """Interpolate position in cylindrical coordinates around a pivot.

    Simultaneously interpolates azimuth, elevation, and radius. Traces a
    spiral when the endpoints differ in azimuth (e.g., descent from
    overhead to an orbit position at a different angle). Azimuth wraps
    via the shortest arc. Returns exact endpoints at t=0 and t=1
    (no float round-trip error).
    """
    if t <= 0:
        return np.asarray(pos_a, np.float32).copy()
    if t >= 1:
        return np.asarray(pos_b, np.float32).copy()
    up = np.asarray(up, np.float64)
    pivot = np.asarray(pivot, np.float64)
    da = np.asarray(pos_a, np.float64) - pivot
    db = np.asarray(pos_b, np.float64) - pivot
    ha = da.copy(); ha[2] = 0
    hb = db.copy(); hb[2] = 0
    na, nb = np.linalg.norm(ha), np.linalg.norm(hb)
    az_a = np.arctan2(da[1], da[0]) if na > 1e-3 else np.arctan2(db[1], db[0])
    az_b = np.arctan2(db[1], db[0]) if nb > 1e-3 else az_a
    daz = az_b - az_a
    if daz > np.pi:
        daz -= 2 * np.pi
    if daz < -np.pi:
        daz += 2 * np.pi
    elev_a = np.arctan2(da[2], max(na, 1e-3))
    elev_b = np.arctan2(db[2], max(nb, 1e-3))
    rad_a = np.linalg.norm(da)
    rad_b = np.linalg.norm(db)
    az = az_a + daz * t
    radius = rad_a * (1 - t) + rad_b * t
    elev = elev_a * (1 - t) + elev_b * t
    horiz = np.array([np.cos(az), np.sin(az), 0.0])
    return (pivot + horiz * (radius * np.cos(elev))
            + up * (radius * np.sin(elev))).astype(np.float32)


def circle_positions(center, radius, up, n, az_start_deg=0.0):
    """Generate n evenly spaced positions on a horizontal circle.

    Returns:
        ndarray shape (n, 3), float32.
    """
    center = np.asarray(center, np.float64)
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    angles = np.radians(az_start_deg) + np.linspace(0, 2 * np.pi, n,
                                                     endpoint=False)
    pts = np.zeros((n, 3), np.float64)
    for i, az in enumerate(angles):
        pts[i] = center + np.array([np.cos(az), np.sin(az), 0.0]) * radius
    return pts.astype(np.float32)
