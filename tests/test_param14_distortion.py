"""Tests for 14-parameter distortion model."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from deltacamera.distortion import (
    distort_points,
    undistort_points,
)


def make_awkward_14_coeffs():
    """Create awkward (non-round) 14-parameter distortion coefficients."""
    return np.array([
        -0.10457,   # k1
        0.03821,    # k2
        0.00173,    # p1
        -0.00289,   # p2
        0.01534,    # k3
        0.00234,    # k4
        -0.00167,   # k5
        0.00089,    # k6
        0.00056,    # s1
        -0.00123,   # s2
        0.00078,    # s3
        -0.00045,   # s4
        0.02345,    # tau_x
        -0.01876,   # tau_y
    ], dtype=np.float64)


def make_12_param_coeffs():
    """Create 12-parameter coefficients (no tilt)."""
    return np.array([
        -0.10457, 0.03821, 0.00173, -0.00289,
        0.01534, 0.00234, -0.00167, 0.00089,
        0.00056, -0.00123, 0.00078, -0.00045,
    ], dtype=np.float64)


class TestDistort14Param:
    """Test 14-parameter distortion."""

    def test_distort_basic(self):
        """Test that distortion runs without error."""
        d = make_awkward_14_coeffs()
        points = np.array([[0.1, 0.2], [-0.15, 0.08], [0.0, 0.0]], dtype=np.float64)

        result = distort_points(points, d)

        assert result.shape == points.shape
        assert not np.any(np.isnan(result))

    def test_origin_unchanged_no_tilt(self):
        """Point at origin should remain at origin (without tilt)."""
        d = make_12_param_coeffs()  # No tilt
        points = np.array([[0.0, 0.0]], dtype=np.float64)

        result = distort_points(points, d)

        # Without tilt, origin stays at origin
        np.testing.assert_allclose(result, [[0.0, 0.0]], atol=1e-10)

    def test_no_tilt_matches_12_param(self):
        """Without tilt, should match 12-param distortion."""
        d12 = make_12_param_coeffs()
        d14 = np.zeros(14, dtype=np.float64)
        d14[:12] = d12

        np.random.seed(42)
        points = np.random.randn(50, 2).astype(np.float64) * 0.3

        result_14 = distort_points(points, d14)

        # Use the existing 12-param implementation for comparison
        from deltacamera.distortion import _distort_points
        # Reorder coefficients to match existing format
        d_reordered = np.zeros(12, dtype=np.float32)
        d_reordered[:12] = d12.astype(np.float32)
        result_12 = _distort_points(
            points.astype(np.float32), d_reordered,
            polar_ud_valid=None, check_validity=False, clip_to_valid=False, dst=None
        )

        np.testing.assert_allclose(result_14, result_12, atol=1e-6)


class TestUndistort14Param:
    """Test 14-parameter undistortion."""

    def test_undistort_basic(self):
        """Test that undistortion runs without error."""
        d = make_awkward_14_coeffs()
        points = np.array([[0.1, 0.2], [-0.15, 0.08]], dtype=np.float64)

        result = undistort_points(points, d)

        assert result.shape == points.shape
        assert not np.any(np.isnan(result))

    def test_roundtrip_12_param(self):
        """Distort then undistort should recover original (12-param, no tilt)."""
        d = make_12_param_coeffs()

        np.random.seed(123)
        points = np.random.randn(30, 2).astype(np.float64) * 0.2

        distorted = distort_points(points, d)
        recovered = undistort_points(distorted, d)

        np.testing.assert_allclose(recovered, points, atol=1e-6)

    def test_roundtrip_14_param(self):
        """Distort then undistort should recover original (14-param with tilt)."""
        d = make_awkward_14_coeffs()

        np.random.seed(456)
        points = np.random.randn(30, 2).astype(np.float64) * 0.15

        distorted = distort_points(points, d)
        recovered = undistort_points(distorted, d)

        np.testing.assert_allclose(recovered, points, atol=1e-6)

    def test_roundtrip_strong_tilt(self):
        """Test roundtrip with stronger tilt values."""
        d = make_awkward_14_coeffs()
        d[12] = 0.08  # stronger tau_x
        d[13] = -0.06  # stronger tau_y

        np.random.seed(789)
        points = np.random.randn(30, 2).astype(np.float64) * 0.1

        distorted = distort_points(points, d)
        recovered = undistort_points(distorted, d)

        np.testing.assert_allclose(recovered, points, atol=1e-5)


class TestTiltTransformations:
    """Test tilt-specific behavior."""

    def test_tilt_only(self):
        """Test distortion with only tilt (no radial/tangential)."""
        d = np.zeros(14, dtype=np.float64)
        d[12] = 0.05  # tau_x
        d[13] = -0.03  # tau_y

        points = np.array([[0.2, 0.1], [-0.1, 0.3]], dtype=np.float64)

        distorted = distort_points(points, d)

        # Points should be different from original
        assert not np.allclose(distorted, points)

        # Roundtrip should work (float32 precision limits accuracy)
        recovered = undistort_points(distorted, d)
        np.testing.assert_allclose(recovered, points, atol=1e-6)

    def test_tilt_rotation_matrix(self):
        """Verify tilt rotation matches expected Ry @ Rx."""
        tau_x = 0.05
        tau_y = -0.03

        # Build rotation manually
        Rx = Rotation.from_euler('x', tau_x).as_matrix()
        Ry = Rotation.from_euler('y', tau_y).as_matrix()
        R = Ry @ Rx

        # Test point
        p = np.array([0.2, 0.15, 1.0])
        p_rot = R @ p
        expected = p_rot[:2] / p_rot[2]

        # Apply through our distortion (with zero radial/tangential)
        d = np.zeros(14, dtype=np.float64)
        d[12] = tau_x
        d[13] = tau_y

        points = np.array([[0.2, 0.15]], dtype=np.float64)
        result = distort_points(points, d)

        # float32 precision limits accuracy to ~1e-7
        np.testing.assert_allclose(result[0], expected, atol=1e-6)


class TestEdgeCases:
    """Test edge cases and special conditions."""

    def test_zero_coefficients(self):
        """All-zero coefficients should be identity."""
        d = np.zeros(14, dtype=np.float64)

        np.random.seed(111)
        points = np.random.randn(20, 2).astype(np.float64) * 0.3

        result = distort_points(points, d)

        np.testing.assert_allclose(result, points, atol=1e-10)

    def test_short_coefficient_array(self):
        """Shorter coefficient arrays should be zero-padded."""
        d5 = np.array([-0.1, 0.05, 0.001, -0.002, 0.01], dtype=np.float64)

        points = np.array([[0.1, 0.2]], dtype=np.float64)

        result = distort_points(points, d5)

        # Should not raise and should produce valid output
        assert result.shape == points.shape
        assert not np.any(np.isnan(result))

    def test_large_tilt(self):
        """Test with larger (but still reasonable) tilt values."""
        d = make_awkward_14_coeffs()
        d[12] = 0.15  # ~8.6 degrees
        d[13] = -0.12  # ~6.9 degrees

        np.random.seed(222)
        points = np.random.randn(20, 2).astype(np.float64) * 0.08

        distorted = distort_points(points, d)
        recovered = undistort_points(distorted, d)

        np.testing.assert_allclose(recovered, points, atol=1e-5)

