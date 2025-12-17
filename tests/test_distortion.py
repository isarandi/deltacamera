"""Tests for distortion and undistortion operations."""

import numpy as np
import pytest

import lensform
import lensform.distortion
import lensform.validity
from conftest import (
    BROWN_CONRADY_COEFFS,
    BROWN_CONRADY_5_COEFFS,
    FISHEYE_COEFFS,
    extend_distortion_coeffs,
    sample_inside_polygon,
    make_camera_with_distortion,
)


# =============================================================================
# Brown-Conrady distortion roundtrip tests
# =============================================================================

class TestBrownConradyRoundtrip:
    """Test that undistort(distort(p)) ≈ p for Brown-Conrady model."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS)
    def test_roundtrip_12_param(self, d):
        """Roundtrip with 12-parameter model."""
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        valid_region_ud, _ = lensform.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )

        # Check if boundary is at r=5 default (no Jacobian=0 crossing)
        # If so, use smaller region to avoid extreme distortion values
        boundary_r = np.max(np.linalg.norm(valid_region_ud, axis=1))
        if boundary_r > 4.5:
            # Practical limit, not real boundary - sample from smaller region
            # Use uniform in [-0.3, 0.3] to ensure bounded radius
            points = (np.random.rand(500, 2).astype(np.float32) - 0.5) * 0.6
        else:
            points = sample_inside_polygon(valid_region_ud, n=500, margin=0.1)

        distorted = lensform.distortion.distort_points(
            points, d, check_validity=False
        )
        recovered = lensform.distortion.undistort_points(
            distorted, d, check_validity=False
        )

        np.testing.assert_allclose(recovered, points, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("d", BROWN_CONRADY_5_COEFFS)
    def test_roundtrip_5_param(self, d):
        """Roundtrip with 5-parameter model (OpenCV default)."""
        d12 = extend_distortion_coeffs(d, 12)
        camera = make_camera_with_distortion(d12)

        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        valid_region_ud, _ = lensform.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )

        # Check if boundary is at r=5 default (no Jacobian=0 crossing)
        boundary_r = np.max(np.linalg.norm(valid_region_ud, axis=1))
        if boundary_r > 4.5:
            # Practical limit - sample from smaller region
            points = (np.random.rand(500, 2).astype(np.float32) - 0.5) * 0.6
        else:
            points = sample_inside_polygon(valid_region_ud, n=500, margin=0.1)

        distorted = lensform.distortion.distort_points(
            points, d12, check_validity=False
        )
        recovered = lensform.distortion.undistort_points(
            distorted, d12, check_validity=False
        )

        np.testing.assert_allclose(recovered, points, rtol=1e-4, atol=1e-6)

    def test_roundtrip_at_origin(self):
        """Point at origin should be unchanged."""
        d = BROWN_CONRADY_COEFFS[0]
        origin = np.array([[0, 0]], np.float32)

        distorted = lensform.distortion.distort_points(
            origin, d, check_validity=False
        )
        np.testing.assert_allclose(distorted, origin, atol=1e-10)

        recovered = lensform.distortion.undistort_points(
            distorted, d, check_validity=False
        )
        np.testing.assert_allclose(recovered, origin, atol=1e-10)

    def test_roundtrip_small_radius(self):
        """Points very close to center should have minimal distortion."""
        d = BROWN_CONRADY_COEFFS[0]
        small_points = np.random.randn(100, 2).astype(np.float32) * 0.01

        distorted = lensform.distortion.distort_points(
            small_points, d, check_validity=False
        )
        # Small points should barely move
        np.testing.assert_allclose(distorted, small_points, rtol=0.01, atol=1e-4)


# =============================================================================
# Fisheye distortion roundtrip tests
# =============================================================================

class TestFisheyeRoundtrip:
    """Test that undistort(distort(p)) ≈ p for fisheye model."""

    @pytest.mark.parametrize("d", FISHEYE_COEFFS)
    def test_roundtrip(self, d):
        """Basic roundtrip test."""
        ru_valid, _ = lensform.validity.fisheye_valid_r_max(d)

        # Handle infinite valid radius (well-behaved distortion)
        ru_max = ru_valid if np.isfinite(ru_valid) else 5.0

        # Sample points inside valid region
        r = np.random.uniform(0, ru_max * 0.8, 500).astype(np.float32)
        theta = np.random.uniform(-np.pi, np.pi, 500).astype(np.float32)
        points = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

        distorted = lensform.distortion.distort_points_fisheye(
            points, d, check_validity=False
        )
        recovered = lensform.distortion.undistort_points_fisheye(
            distorted, d, check_validity=False
        )

        # Fisheye undistortion has lower precision due to iterative algorithm
        # Allow ~3% relative error for strong distortions
        np.testing.assert_allclose(recovered, points, rtol=3e-2, atol=1e-4)

    def test_roundtrip_at_origin(self):
        """Point at origin should be unchanged."""
        d = FISHEYE_COEFFS[0]
        origin = np.array([[0, 0]], np.float32)

        distorted = lensform.distortion.distort_points_fisheye(
            origin, d, check_validity=False
        )
        np.testing.assert_allclose(distorted, origin, atol=1e-10)


# =============================================================================
# Validity checking tests
# =============================================================================

class TestDistortionValidityChecking:
    """Test that validity checking works correctly."""

    def test_valid_points_not_nan(self):
        """Points inside valid region should not produce NaN."""
        d = BROWN_CONRADY_COEFFS[0]
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        valid_region_ud, _ = lensform.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )
        points = sample_inside_polygon(valid_region_ud, n=200, margin=0.15)

        result = lensform.distortion.distort_points(
            points, d, check_validity=True
        )

        assert not np.any(np.isnan(result)), "Valid points should not produce NaN"

    def test_invalid_points_are_nan(self):
        """Points outside valid region should produce NaN."""
        d = BROWN_CONRADY_COEFFS[0]
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        valid_region_ud, _ = lensform.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )

        # Scale points to be outside valid region
        outside_points = valid_region_ud[::4] * 1.5  # Subsample and scale

        result = lensform.distortion.distort_points(
            outside_points, d, check_validity=True
        )

        nan_count = np.sum(np.isnan(result[:, 0]))
        # Most points should be NaN
        assert nan_count > len(outside_points) * 0.5, \
            f"Expected most points to be NaN, got {nan_count}/{len(outside_points)}"


# =============================================================================
# Zero distortion tests
# =============================================================================

class TestZeroDistortion:
    """Test behavior with zero distortion coefficients."""

    def test_zero_distortion_is_identity(self):
        """Zero distortion coefficients should not change points."""
        d = np.zeros(12, np.float32)
        points = np.random.randn(100, 2).astype(np.float32)

        distorted = lensform.distortion.distort_points(
            points, d, check_validity=False
        )

        np.testing.assert_allclose(distorted, points, atol=1e-10)

    def test_zero_fisheye_roundtrip(self):
        """Zero fisheye coefficients roundtrip should work."""
        d = np.zeros(4, np.float32)
        points = np.random.randn(100, 2).astype(np.float32) * 0.5

        distorted = lensform.distortion.distort_points_fisheye(
            points, d, check_validity=False
        )
        recovered = lensform.distortion.undistort_points_fisheye(
            distorted, d, check_validity=False
        )

        np.testing.assert_allclose(recovered, points, rtol=1e-4, atol=1e-6)


# =============================================================================
# Edge cases
# =============================================================================

class TestDistortionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_points(self):
        """Empty point arrays should not crash."""
        d = BROWN_CONRADY_COEFFS[0]
        empty = np.zeros((0, 2), np.float32)

        result = lensform.distortion.distort_points(empty, d, check_validity=False)
        assert result.shape == (0, 2)

    def test_single_point(self):
        """Single point should work."""
        d = BROWN_CONRADY_COEFFS[0]
        point = np.array([[0.1, 0.1]], np.float32)

        distorted = lensform.distortion.distort_points(
            point, d, check_validity=False
        )
        recovered = lensform.distortion.undistort_points(
            distorted, d, check_validity=False
        )

        np.testing.assert_allclose(recovered, point, rtol=1e-4)

    def test_symmetric_distortion(self):
        """Symmetric points should distort symmetrically (no tangential)."""
        d = extend_distortion_coeffs(np.array([-0.2, 0.1, 0, 0, 0], np.float32))
        points = np.array([
            [0.3, 0.0],
            [-0.3, 0.0],
            [0.0, 0.3],
            [0.0, -0.3],
        ], np.float32)

        distorted = lensform.distortion.distort_points(
            points, d, check_validity=False
        )

        # All points at same radius should distort to same radius
        r_distorted = np.linalg.norm(distorted, axis=1)
        np.testing.assert_allclose(r_distorted, r_distorted[0], rtol=1e-6)
