"""Camera-to-camera interpolation and departure blending."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from deltacamera import Camera
from .orientation import upright_rotation
from .easing import smoothstep


def slerp_cameras(cam_a, cam_b, t, ease=None):
    """Spherical interpolation between two cameras.

    Rotation: Slerp. Position: linear. Intrinsics: linear.
    Exact at endpoints (t=0 gives cam_a, t=1 gives cam_b).
    """
    if ease is not None:
        t = ease(t)
    Ka = cam_a.intrinsic_matrix.astype(np.float32)
    Kb = cam_b.intrinsic_matrix.astype(np.float32)
    K = Ka * (1 - t) + Kb * t
    Ra = cam_a.get_extrinsic_matrix()[:3, :3]
    Rb = cam_b.get_extrinsic_matrix()[:3, :3]
    R = Slerp([0, 1], Rotation.from_matrix(
        np.stack([Ra, Rb])))(t).as_matrix().astype(np.float32)
    C = (cam_a.t * (1 - t) + cam_b.t * t).astype(np.float32)
    ext = np.eye(4, dtype=np.float32)
    ext[:3, :3] = R
    ext[:3, 3] = -R @ C
    return Camera(
        extrinsic_matrix=ext, intrinsic_matrix=K,
        world_up=cam_a.world_up,
        image_shape=cam_a.image_shape or cam_b.image_shape)


def interpolate_cameras(cam_a, cam_b, t, focal_space='log'):
    """Interpolate between two cameras with log-space focal length.

    Orientation is interpolated via target-point direction (avoids gimbal
    lock). Focal length is interpolated in log-space (perceptually
    uniform zoom). Position is linear.
    """
    f_a = np.array([cam_a.intrinsic_matrix[0, 0],
                     cam_a.intrinsic_matrix[1, 1]], np.float64)
    f_b = np.array([cam_b.intrinsic_matrix[0, 0],
                     cam_b.intrinsic_matrix[1, 1]], np.float64)
    c_a = cam_a.intrinsic_matrix[:2, 2].astype(np.float64)
    c_b = cam_b.intrinsic_matrix[:2, 2].astype(np.float64)
    if focal_space == 'log':
        f = np.exp(np.log(f_a) + t * (np.log(f_b) - np.log(f_a)))
        c = np.exp(np.log(c_a) + t * (np.log(c_b) - np.log(c_a)))
    else:
        f = f_a * (1 - t) + f_b * t
        c = c_a * (1 - t) + c_b * t
    K = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], np.float32)
    C = (cam_a.t * (1 - t) + cam_b.t * t).astype(np.float32)
    target_a = cam_a.t + cam_a.R[2] * 100
    target_b = cam_b.t + cam_b.R[2] * 100
    target = target_a * (1 - t) + target_b * t
    cam = Camera(intrinsic_matrix=K, world_up=cam_a.world_up,
                 image_shape=cam_a.image_shape or cam_b.image_shape)
    return cam.copy(optical_center=C).turned_towards(
        target_world_point=target.astype(np.float32))


class DepartureBlend:
    """Smooth transition from an exact camera to an upright orientation.

    Fades the rig camera's roll correction (Slerp to identity) and
    intrinsic matrix (lerp to target_K) over n_steps.
    """

    def __init__(self, source_cam, target_K, n_steps, *,
                 world_up=None, target=None, ease=None):
        self._n_steps = n_steps
        self._ease = ease or smoothstep
        self._target_K = np.asarray(target_K, np.float32)
        self._source_K = source_cam.intrinsic_matrix.astype(np.float32)
        R_exact = source_cam.get_extrinsic_matrix()[:3, :3].astype(np.float64)
        R_upright, _ = upright_rotation(source_cam.t, target, world_up)
        self._correction = R_exact @ R_upright.astype(np.float64).T
        self._count = 0

    def apply(self, R_upright, step=None):
        """Apply the blend at the given step.

        Args:
            R_upright: upright rotation at the current position (3, 3).
            step: 1-based step index. If None, uses internal counter.

        Returns:
            R: blended rotation matrix (3, 3), float32.
            K: blended intrinsic matrix (3, 3), float32.
        """
        if step is None:
            self._count += 1
            step = self._count
        t = self._ease(step / self._n_steps)
        corr = Slerp([0, 1], Rotation.from_matrix(np.stack([
            self._correction, np.eye(3)
        ])))(t).as_matrix()
        R = (corr @ R_upright.astype(np.float64)).astype(np.float32)
        K = self._source_K * (1 - t) + self._target_K * t
        return R, K

    @property
    def active(self):
        return self._count < self._n_steps

    @property
    def n_steps(self):
        return self._n_steps
