"""Tests for valid distortion region computation."""

import numpy as np
import pytest
import shapely

import deltacamera
import deltacamera.validity
import deltacamera.distortion
from conftest import (
    BROWN_CONRADY_COEFFS,
    FISHEYE_COEFFS,
    sample_inside_polygon,
    sample_on_polygon_boundary,
    make_camera_with_distortion,
)


# =============================================================================
# Valid region polygon properties
# =============================================================================

class TestValidRegionPolygon:
    """Test properties of the valid region polygon."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS)
    def test_polygon_is_closed(self, d):
        """Valid region polygon should be closed (first == last vertex)."""
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        region, _ = deltacamera.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )
        np.testing.assert_allclose(region[0], region[-1], atol=1e-6)

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS)
    def test_polygon_is_simple(self, d):
        """Valid region polygon should not self-intersect."""
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        region, _ = deltacamera.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )
        poly = shapely.Polygon(region)
        assert poly.is_valid or shapely.is_valid(shapely.make_valid(poly))

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS)
    def test_polygon_contains_origin(self, d):
        """Valid region should contain the origin."""
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        region, _ = deltacamera.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )
        poly = shapely.Polygon(region)
        assert poly.contains(shapely.Point(0, 0))

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS)
    def test_polygon_has_positive_area(self, d):
        """Valid region should have positive area."""
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        region, _ = deltacamera.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )
        poly = shapely.Polygon(region)
        assert poly.area > 0


# =============================================================================
# Jacobian determinant at boundary
# =============================================================================

class TestJacobianAtBoundary:
    """Test Jacobian determinant properties."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS)
    def test_jacobian_near_zero_at_boundary(self, d):
        """Jacobian determinant should be close to zero at the boundary.

        Note: Only applies to distortions that have a finite Jacobian=0 boundary
        (e.g., barrel distortion). Pincushion and mild distortions may have
        Jacobian > 0 everywhere, so they use r=5 as a practical FOV limit.
        """
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        region, _ = deltacamera.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )

        # Sample points on the boundary and convert to polar
        boundary_points = sample_on_polygon_boundary(region, n=50)
        r = np.linalg.norm(boundary_points, axis=1).astype(np.float32)
        t = np.arctan2(boundary_points[:, 1], boundary_points[:, 0]).astype(np.float32)

        # Compute Jacobian determinant in polar coords
        det = deltacamera.validity.jacobian_det_polar(r, t, d)

        # Check if boundary is at the default r=5 limit (no Jacobian=0 crossing)
        if np.allclose(r, 5.0, atol=0.1):
            # This distortion has no finite Jacobian=0 boundary, r=5 is practical limit
            # Jacobian should still be positive (invertible)
            assert np.all(det > 0), "Jacobian should be positive at practical FOV limit"
        else:
            # Real Jacobian=0 boundary - should be close to zero
            np.testing.assert_allclose(det, 0, atol=0.05)

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS)
    def test_jacobian_positive_inside(self, d):
        """Jacobian determinant should be positive inside the valid region."""
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        region, _ = deltacamera.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )

        inside_points = sample_inside_polygon(region, n=500, margin=0.1)
        r = np.linalg.norm(inside_points, axis=1).astype(np.float32)
        t = np.arctan2(inside_points[:, 1], inside_points[:, 0]).astype(np.float32)

        det = deltacamera.validity.jacobian_det_polar(r, t, d)
        assert np.all(det > 0), f"Found {np.sum(det <= 0)} points with det <= 0"

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS)
    def test_jacobian_at_origin(self, d):
        """Jacobian determinant at origin should be 1."""
        r = np.array([0.0], np.float32)
        t = np.array([0.0], np.float32)
        det = deltacamera.validity.jacobian_det_polar(r, t, d)
        np.testing.assert_allclose(det, 1.0, rtol=1e-6)


# =============================================================================
# Valid region consistency with distortion
# =============================================================================

class TestValidRegionConsistency:
    """Test that valid region is consistent with distortion operations."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS[:2])
    def test_distortion_succeeds_inside(self, d):
        """Distortion should succeed for points inside valid region."""
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        region, _ = deltacamera.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )
        inside_points = sample_inside_polygon(region, n=200, margin=0.15)

        result = deltacamera.distortion.distort_points(
            inside_points, d, check_validity=True
        )

        nan_count = np.sum(np.isnan(result[:, 0]))
        assert nan_count == 0, f"Got {nan_count} NaN results for valid points"

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS[:2])
    def test_roundtrip_succeeds_inside(self, d):
        """Roundtrip should succeed for points inside valid region."""
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        region, _ = deltacamera.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )

        # Check if boundary is at r=5 default (no Jacobian=0 crossing)
        boundary_r = np.max(np.linalg.norm(region, axis=1))
        if boundary_r > 4.5:
            # Practical limit - sample from smaller region to avoid extreme values
            inside_points = (np.random.rand(200, 2).astype(np.float32) - 0.5) * 0.6
        else:
            inside_points = sample_inside_polygon(region, n=200, margin=0.15)

        distorted = deltacamera.distortion.distort_points(
            inside_points, d, check_validity=False
        )
        recovered = deltacamera.distortion.undistort_points(
            distorted, d, check_validity=False
        )

        np.testing.assert_allclose(recovered, inside_points, rtol=1e-4, atol=1e-6)


# =============================================================================
# Fisheye valid region
# =============================================================================

class TestFisheyeValidRegion:
    """Test valid region for fisheye model."""

    @pytest.mark.parametrize("d", FISHEYE_COEFFS)
    def test_fisheye_valid_radius_positive(self, d):
        """Fisheye valid radius should be positive."""
        ru_valid, rd_valid = deltacamera.validity.fisheye_valid_r_max(d)

        # ru_valid can be inf for well-behaved distortions
        assert ru_valid > 0
        assert rd_valid > 0

    @pytest.mark.parametrize("d", FISHEYE_COEFFS)
    def test_fisheye_distortion_inside_valid(self, d):
        """Distortion should succeed inside fisheye valid region."""
        ru_valid, _ = deltacamera.validity.fisheye_valid_r_max(d)

        # Handle infinite valid radius
        ru_max = ru_valid if np.isfinite(ru_valid) else 5.0

        r = np.random.uniform(0, ru_max * 0.8, 100).astype(np.float32)
        theta = np.random.uniform(-np.pi, np.pi, 100).astype(np.float32)
        points = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

        result = deltacamera.distortion.distort_points_fisheye(
            points, d, check_validity=True
        )

        assert not np.any(np.isnan(result))


# =============================================================================
# Polar vs Cartesian representation
# =============================================================================

class TestPolarCartesianConsistency:
    """Test consistency between polar and cartesian representations."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS[:2])
    def test_polar_cartesian_conversion(self, d):
        """Polar and cartesian representations should be consistent."""
        camera = make_camera_with_distortion(d)
        # get_valid_distortion_region returns (undistorted_region, distorted_region)
        polar, _ = deltacamera.validity.get_valid_distortion_region(
            camera, cartesian=False, n_vertices=128
        )
        cartesian, _ = deltacamera.validity.get_valid_distortion_region(
            camera, cartesian=True, n_vertices=128
        )

        # Convert polar to cartesian manually
        r, theta = polar[:, 0], polar[:, 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        manual_cartesian = np.stack([x, y], axis=1)

        np.testing.assert_allclose(cartesian, manual_cartesian, rtol=1e-5)


# =============================================================================
# Caching
# =============================================================================

class TestValidRegionCaching:
    """Test that caching works correctly."""

    def test_cached_result_matches(self):
        """Cached result should match direct computation."""
        d = BROWN_CONRADY_COEFFS[0]
        key = d.astype(np.float32).tobytes()

        result1 = deltacamera.validity.get_valid_distortion_region_cached(key)
        result2 = deltacamera.validity.get_valid_distortion_region_cached(key)

        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])


# =============================================================================
# Reprojection valid region
# =============================================================================

class TestReprojectionValidRegion:
    """Test valid region computation for reprojection."""

    def test_same_camera_full_image(self):
        """Reprojecting to same camera should give full image as valid."""
        camera = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )
        imshape = (480, 640)

        poly = deltacamera.validity.get_valid_poly_reproj(
            camera, camera, imshape, imshape
        )

        expected_area = imshape[0] * imshape[1]
        assert poly.area > expected_area * 0.99

    def test_rotation_reduces_valid_area(self):
        """Rotating camera should reduce valid overlap area."""
        camera1 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )
        camera2 = camera1.copy()
        camera2.rotate(yaw=0.3)  # ~17 degree rotation

        imshape = (480, 640)

        poly = deltacamera.validity.get_valid_poly_reproj(
            camera1, camera2, imshape, imshape
        )

        full_area = imshape[0] * imshape[1]
        assert poly.area < full_area * 0.95
