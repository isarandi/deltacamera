"""Intrinsic camera parameter estimation utilities."""

import numpy as np


def estimate_fov_from_vanishing_points(vp1, vp2, imshape):
    """Estimate camera FOV from two orthogonal vanishing points.

    Given two vanishing points corresponding to orthogonal directions in 3D space,
    estimates the camera's field of view assuming the principal point is at the
    image center.

    Args:
        vp1: First vanishing point as (x, y) in pixel coordinates.
        vp2: Second vanishing point as (x, y) in pixel coordinates.
        imshape: Image shape as (height, width) or (height, width, channels).

    Returns:
        Dictionary with keys:
            - 'f': Focal length in pixels
            - 'fov_h': Horizontal field of view in degrees
            - 'fov_v': Vertical field of view in degrees
            - 'fov_d': Diagonal field of view in degrees
            - 'K': 3x3 intrinsic matrix

        Returns None if the vanishing points don't satisfy the orthogonality
        constraint (i.e., they would imply an imaginary focal length).
    """
    vp1 = np.asarray(vp1, dtype=np.float64)
    vp2 = np.asarray(vp2, dtype=np.float64)

    height, width = imshape[:2]
    c = np.array([width / 2, height / 2])

    # For orthogonal vanishing points: (v1-c)·(v2-c) = -f²
    dot = (vp1 - c) @ (vp2 - c)

    if dot >= 0:
        return None

    f = np.sqrt(-dot)

    fov_h = 2 * np.rad2deg(np.arctan(width / (2 * f)))
    fov_v = 2 * np.rad2deg(np.arctan(height / (2 * f)))
    fov_d = 2 * np.rad2deg(np.arctan(np.sqrt(width**2 + height**2) / (2 * f)))

    K = np.array([
        [f, 0, c[0]],
        [0, f, c[1]],
        [0, 0, 1]
    ])

    return {
        'f': f,
        'fov_h': fov_h,
        'fov_v': fov_v,
        'fov_d': fov_d,
        'K': K,
    }
