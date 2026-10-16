"""Tests for deltacamera.trajectory subpackage."""

import numpy as np
import pytest

from deltacamera import Camera
from deltacamera.trajectory import (
    smoothstep, smootherstep, ramp,
    lerp, cylindrical_arc, circle_positions,
    upright_rotation, UprightTracker,
    slerp_cameras, interpolate_cameras, DepartureBlend,
    smooth_cameras, trapezoidal_kernel, moving_average_1d,
    TrackingCamera,
)


class TestEasing:
    def test_smoothstep_endpoints(self):
        assert smoothstep(0.0) == 0.0
        assert smoothstep(1.0) == 1.0

    def test_smoothstep_midpoint(self):
        assert smoothstep(0.5) == pytest.approx(0.5)

    def test_smoothstep_clamping(self):
        assert smoothstep(-0.5) == 0.0
        assert smoothstep(1.5) == 1.0

    def test_smoothstep_monotonic(self):
        t = np.linspace(0, 1, 100)
        vals = smoothstep(t)
        assert np.all(np.diff(vals) >= 0)

    def test_smootherstep_endpoints(self):
        assert smootherstep(0.0) == 0.0
        assert smootherstep(1.0) == 1.0

    def test_smootherstep_midpoint(self):
        assert smootherstep(0.5) == pytest.approx(0.5)

    def test_smootherstep_c2(self):
        t = np.linspace(0, 1, 10000)
        vals = smootherstep(t)
        d2 = np.diff(vals, 2) / (t[1] - t[0]) ** 2
        assert abs(d2[0]) < 5.0
        assert abs(d2[-1]) < 5.0

    def test_ramp_before_after(self):
        assert ramp(0, 10, 20) == 0.0
        assert ramp(50, 10, 20) == 1.0

    def test_ramp_with_smootherstep(self):
        val = ramp(20, 10, 20, ease=smootherstep)
        assert 0 < val < 1

    def test_smoothstep_array(self):
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = smoothstep(t)
        assert result.shape == (5,)


class TestPaths:
    def test_lerp_endpoints(self):
        a = np.array([1, 2, 3], np.float32)
        b = np.array([4, 5, 6], np.float32)
        np.testing.assert_allclose(lerp(a, b, 0), a, atol=1e-6)
        np.testing.assert_allclose(lerp(a, b, 1), b, atol=1e-6)

    def test_cylindrical_arc_endpoints(self):
        pivot = np.array([0, 0, 0], np.float64)
        up = np.array([0, 0, 1], np.float64)
        a = np.array([3000, 0, 1500], np.float64)
        b = np.array([0, 3000, 4000], np.float64)
        np.testing.assert_allclose(
            cylindrical_arc(a, b, 0.0, pivot, up), a, atol=1.0)
        np.testing.assert_allclose(
            cylindrical_arc(a, b, 1.0, pivot, up), b, atol=1.0)

    def test_cylindrical_arc_preserves_radius(self):
        pivot = np.zeros(3)
        up = np.array([0, 0, 1.0])
        r = 5000.0
        a = np.array([r, 0, 0])
        b = np.array([0, r, 0])
        mid = cylindrical_arc(a, b, 0.5, pivot, up)
        assert abs(np.linalg.norm(mid - pivot) - r) < 1.0

    def test_cylindrical_arc_azimuth_wrapping(self):
        pivot = np.zeros(3)
        up = np.array([0, 0, 1.0])
        a = np.array([5000, 100, 0])   # near az=0
        b = np.array([5000, -100, 0])  # near az=0 from the other side
        mid = cylindrical_arc(a, b, 0.5, pivot, up)
        assert mid[0] > 4000  # should stay near x axis, not go to -x

    def test_paths_follow_up_vector(self):
        """A scene rotated so that up becomes -y (deltacamera's default world_up)
        yields the rotated path: azimuth and elevation are relative to ``up``."""
        # Rotation taking +z to -y
        rot = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
        pivot = np.array([10.0, 20, 30])
        a = np.array([3000.0, 500, 1500])
        b = np.array([-200.0, 3000, 4000])
        z_up = np.array([0, 0, 1.0])
        for t in (0.25, 0.5, 0.8):
            ref = cylindrical_arc(a, b, t, pivot, z_up)
            rotated = cylindrical_arc(rot @ a, rot @ b, t, rot @ pivot, rot @ z_up)
            mid_z = ref - pivot
            mid_y = rotated - rot @ pivot
            # Same radius and height above the horizontal plane
            np.testing.assert_allclose(np.linalg.norm(mid_y), np.linalg.norm(mid_z), rtol=1e-5)
            np.testing.assert_allclose(mid_y @ (rot @ z_up), mid_z @ z_up, atol=1e-2)
        pts = circle_positions(pivot, 1000, [0, -1, 0], 8)
        np.testing.assert_allclose(pts[:, 1], pivot[1], atol=1e-3)  # in the plane y = const
        np.testing.assert_allclose(np.linalg.norm(pts - pivot, axis=1), 1000, atol=1e-2)

    def test_circle_positions_count(self):
        pts = circle_positions([0, 0, 0], 1000, [0, 0, 1], 8)
        assert pts.shape == (8, 3)
        dists = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(dists, 1000, atol=1.0)


class TestOrientation:
    def test_upright_looks_at_target(self):
        R, _ = upright_rotation([0, 0, 5000], [0, 0, 0], [0, 0, 1])
        fwd = R[2]
        expected = np.array([0, 0, -1])
        np.testing.assert_allclose(fwd, expected, atol=1e-4)

    def test_upright_zero_roll(self):
        R, _ = upright_rotation([3000, 0, 1500], [0, 0, 0], [0, 0, 1])
        assert abs(R[0, 2]) < 1e-4

    def test_upright_pole_with_hint(self):
        hint = np.array([1.0, 0.0, 0.0])
        R, right = upright_rotation([0, 0, 5000], [0, 0, 0], [0, 0, 1],
                                     prev_right=hint)
        np.testing.assert_allclose(right, hint, atol=1e-4)

    def test_upright_tracker_continuity(self):
        tracker = UprightTracker(target=[0, 0, 0], world_up=[0, 0, 1])
        positions = []
        for az in np.linspace(0, np.pi, 50):
            elev = np.pi / 2 * (1 - abs(az - np.pi / 2) / (np.pi / 2))
            pos = np.array([np.cos(az) * 3000 * np.cos(elev),
                            np.sin(az) * 3000 * np.cos(elev),
                            3000 * np.sin(elev)])
            R = tracker(pos)
            positions.append(R[0].copy())
        diffs = [np.linalg.norm(positions[i + 1] - positions[i])
                 for i in range(len(positions) - 1)]
        assert max(diffs) < 0.5


class TestInterpolation:
    def _make_cam(self, pos, target):
        return Camera.looking_at(target=target, position=pos,
                                 fov_degrees=55, imshape=(1080, 1920),
                                 world_up=(0, 0, 1))

    def test_slerp_endpoints(self):
        ca = self._make_cam([3000, 0, 1500], [0, 0, 0])
        cb = self._make_cam([0, 3000, 1500], [0, 0, 0])
        r0 = slerp_cameras(ca, cb, 0.0)
        r1 = slerp_cameras(ca, cb, 1.0)
        np.testing.assert_allclose(r0.t, ca.t, atol=1e-3)
        np.testing.assert_allclose(r1.t, cb.t, atol=1e-3)
        np.testing.assert_allclose(r0.R, ca.R, atol=1e-3)
        np.testing.assert_allclose(r1.R, cb.R, atol=1e-3)

    def test_slerp_valid_rotation(self):
        ca = self._make_cam([3000, 0, 1500], [0, 0, 0])
        cb = self._make_cam([0, 3000, 1500], [0, 0, 0])
        mid = slerp_cameras(ca, cb, 0.5)
        R = mid.R
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)
        assert abs(np.linalg.det(R) - 1) < 1e-4

    def test_interpolate_cameras_log_focal(self):
        ca = self._make_cam([3000, 0, 1500], [0, 0, 0])
        cb = ca.zoomed(2.0)
        mid = interpolate_cameras(ca, cb, 0.5, focal_space='log')
        fa = ca.intrinsic_matrix[0, 0]
        fb = cb.intrinsic_matrix[0, 0]
        expected = np.sqrt(fa * fb)
        assert abs(mid.intrinsic_matrix[0, 0] - expected) < 1.0

    def test_departure_blend_ends_at_identity(self):
        cam = self._make_cam([3000, 0, 1500], [0, 0, 0])
        K_target = cam.intrinsic_matrix.copy()
        K_target[0, 0] *= 0.8
        blend = DepartureBlend(cam, K_target, n_steps=10,
                               world_up=[0, 0, 1], target=[0, 0, 0])
        R_up, _ = upright_rotation(cam.t, [0, 0, 0], [0, 0, 1])
        R_final, K_final = blend.apply(R_up, step=10)
        np.testing.assert_allclose(K_final, K_target, atol=1e-3)


class TestSmoothing:
    def test_trapezoidal_kernel_shape(self):
        k = trapezoidal_kernel(15, 41)
        assert len(k) == 15 + 41 + 15

    def test_smooth_constant_unchanged(self):
        cam = Camera.looking_at([0, 0, 0], [0, 0, 5000], 55,
                                (1080, 1920), (0, 0, 1))
        cams = [cam] * 100
        smoothed = smooth_cameras(cams, ramp_frames=5, flat_frames=11)
        for s in smoothed:
            np.testing.assert_allclose(s.t, cam.t, atol=1.0)

    def test_smooth_preserves_length(self):
        cam = Camera.looking_at([0, 0, 0], [0, 0, 5000], 55,
                                (1080, 1920), (0, 0, 1))
        cams = [cam] * 50
        assert len(smooth_cameras(cams)) == 50

    def test_moving_average_constant(self):
        signal = np.ones((20, 3))
        kernel = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        result = moving_average_1d(signal, kernel)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


class TestTracking:
    def test_stationary_target(self):
        cam = Camera.looking_at([0, 0, 0], [0, 0, 5000], 55,
                                (1080, 1920), (0, 0, 1))
        tracker = TrackingCamera(cam, stiffness=0.1, inertia=0.5)
        for _ in range(100):
            cam = tracker.step([0, 0, 0], dt=1 / 30)
        proj = cam.world_to_image(np.array([[0, 0, 0]], np.float32))[0]
        assert abs(proj[0] - 960) < 50
        assert abs(proj[1] - 540) < 50

    def test_dead_zone(self):
        cam = Camera.looking_at([0, 0, 0], [0, 0, 5000], 55,
                                (1080, 1920), (0, 0, 1))
        tracker = TrackingCamera(cam, dead_zone=0.3, stiffness=0.1)
        cam_before = tracker.camera
        tracker.step([10, 0, 0], dt=1 / 30)
        cam_after = tracker.camera
        assert np.allclose(cam_before.R, cam_after.R, atol=1e-4)

    def test_reset(self):
        cam = Camera.looking_at([0, 0, 0], [0, 0, 5000], 55,
                                (1080, 1920), (0, 0, 1))
        tracker = TrackingCamera(cam)
        tracker.step([1000, 0, 0], dt=1 / 30)
        tracker.reset()
        assert tracker._yaw_vel == 0.0
        assert tracker._pitch_vel == 0.0
