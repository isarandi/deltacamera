"""Upright camera orientation with pole-singularity continuity."""

import numpy as np


def upright_rotation(position, target, world_up, prev_right=None):
    """Compute an upright rotation matrix looking from position at target.

    The image x-axis lies in the horizontal plane (zero roll). At the
    pole (looking straight along world_up), prev_right provides azimuth
    continuity instead of picking an arbitrary perpendicular.

    Returns:
        R: rotation matrix (world-to-camera), shape (3, 3), float32.
        right: the computed right vector, for feeding back as prev_right.
    """
    fwd = np.asarray(target, np.float64) - np.asarray(position, np.float64)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    up = np.asarray(world_up, np.float64)
    right = np.cross(fwd, up)
    rn = np.linalg.norm(right)
    if rn < 1e-4 and prev_right is not None:
        right = np.asarray(prev_right, np.float64)
        right = right - fwd * np.dot(right, fwd)
        rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
        rn = 1.0
    right = right / rn
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd]).astype(np.float32)
    return R, right.copy()


class UprightTracker:
    """Stateful upright orientation with pole-singularity continuity.

    Tracks the previous frame's right vector so that azimuth is
    continuous when the camera passes through or near the pole.

    Args:
        target: look-at target, shape (3,).
        world_up: world up direction, shape (3,).
    """

    def __init__(self, target, world_up):
        self._target = np.asarray(target, np.float64)
        self._world_up = np.asarray(world_up, np.float64)
        self._prev_right = None

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = np.asarray(value, np.float64)

    def __call__(self, position):
        """Compute rotation matrix at position, tracking prev_right."""
        R, self._prev_right = upright_rotation(
            position, self._target, self._world_up, self._prev_right)
        return R

    def set_hint(self, camera_or_right):
        """Set the right-vector hint from a Camera or explicit vector.

        Call this after visiting an exact rig camera to maintain azimuth
        continuity when departing.
        """
        if hasattr(camera_or_right, 'R'):
            self._prev_right = np.asarray(camera_or_right.R[0],
                                          np.float64).copy()
        else:
            self._prev_right = np.asarray(camera_or_right, np.float64).copy()

    def reset(self):
        """Clear the right-vector hint."""
        self._prev_right = None
