"""Tests for previously untested functionality.

These tests cover functions exported in __init__.py that had zero test coverage,
including the get_valid_poly bug fix and complex reprojection operations.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import deltacamera
from deltacamera import Camera


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cam_pinhole():
    """Simple pinhole camera without distortion."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    return Camera(intrinsic_matrix=K)


@pytest.fixture
def cam_distorted():
    """Camera with Brown-Conrady distortion (stronger to have visible valid region)."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    # Stronger distortion to have a clearly bounded valid region
    d = np.array([0.3, -0.5, 0.001, 0.001, 0.2, 0.1, 0.05, 0.02], dtype=np.float32)
    return Camera(intrinsic_matrix=K, distortion_coeffs=d)


@pytest.fixture
def cam_distorted_14param():
    """Camera with 14-param Brown-Conrady distortion including tilt."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    d = np.zeros(14, dtype=np.float32)
    d[0] = 0.3   # k1
    d[1] = -0.5  # k2
    d[4] = 0.2   # k4
    d[12] = 0.01  # tau_x
    d[13] = 0.02  # tau_y
    return Camera(intrinsic_matrix=K, distortion_coeffs=d)


@pytest.fixture
def cam_fisheye():
    """Camera with fisheye (Kannala-Brandt) distortion."""
    K = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]], dtype=np.float32)
    d = np.array([0.1, 0.01, 0.001, 0.0001], dtype=np.float32)
    return Camera(intrinsic_matrix=K, distortion_coeffs=d)


@pytest.fixture
def cam_rotated(cam_pinhole):
    """Pinhole camera rotated slightly."""
    # rotate() takes yaw, pitch, roll in radians
    return cam_pinhole.rotate(
        yaw=np.radians(10),
        pitch=np.radians(5),
        roll=np.radians(3)
    )


# =============================================================================
# 1. validity.py - get_valid_poly (bug fix verification)
# =============================================================================


class TestValidityPoly:
    """Tests for get_valid_poly and get_valid_mask - previously untested."""

    def test_get_valid_poly_with_distortion(self, cam_distorted):
        """Exercise get_valid_poly for camera with Brown-Conrady distortion."""
        from shapely.geometry import Polygon, MultiPolygon

        poly = deltacamera.validity.get_valid_poly(cam_distorted, imshape=(480, 640))

        assert isinstance(poly, (Polygon, MultiPolygon))
        assert poly.area > 0

    def test_get_valid_poly_14param_tilt(self, cam_distorted_14param):
        """get_valid_poly should work with 14-param model including tilt."""
        from shapely.geometry import Polygon, MultiPolygon

        poly = deltacamera.validity.get_valid_poly(cam_distorted_14param, imshape=(480, 640))

        assert isinstance(poly, (Polygon, MultiPolygon))
        assert poly.area > 0

    def test_get_valid_poly_no_distortion(self, cam_pinhole):
        """get_valid_poly for undistorted camera should cover full image."""
        from shapely.geometry import Polygon, MultiPolygon

        poly = deltacamera.validity.get_valid_poly(cam_pinhole, imshape=(480, 640))

        assert isinstance(poly, (Polygon, MultiPolygon))
        # Should cover essentially the full image
        assert poly.area > 480 * 640 * 0.99

    def test_get_valid_mask_returns_rle(self, cam_distorted):
        """get_valid_mask should return RLEMask."""
        from rlemasklib import RLEMask

        mask = deltacamera.get_valid_mask(cam_distorted, imshape=(480, 640))

        assert isinstance(mask, RLEMask)
        assert mask.area() > 0

    def test_get_valid_mask_matches_poly_area(self, cam_distorted):
        """get_valid_mask RLE area should roughly match get_valid_poly area."""
        mask_rle = deltacamera.get_valid_mask(cam_distorted, imshape=(480, 640))
        poly = deltacamera.validity.get_valid_poly(cam_distorted, imshape=(480, 640))

        # Areas should match within 10%
        relative_diff = abs(mask_rle.area() - poly.area) / max(poly.area, 1)
        assert relative_diff < 0.10


# =============================================================================
# 2. reprojection.py - Box functions
# =============================================================================


class TestReprojectBox:
    """Tests for reproject_box and variants - previously untested."""

    def test_reproject_box_identity(self, cam_pinhole):
        """Reprojecting box to same camera should preserve it."""
        import boxlib
        box = np.array([100, 100, 100, 100], dtype=np.float32)  # xywh format

        box_reproj = deltacamera.reproject_box(box, cam_pinhole, cam_pinhole)

        np.testing.assert_allclose(box_reproj, box, atol=1)

    def test_reproject_box_small_rotation(self, cam_pinhole, cam_rotated):
        """Reproject box with small rotation should give valid result."""
        box = np.array([100, 100, 100, 100], dtype=np.float32)  # xywh

        box_reproj = deltacamera.reproject_box(box, cam_pinhole, cam_rotated)

        # Box should have positive width and height
        assert box_reproj[2] > 0  # width
        assert box_reproj[3] > 0  # height

    def test_reproject_box_corners(self, cam_pinhole, cam_rotated):
        """reproject_box_corners should return valid box."""
        box = np.array([100, 100, 100, 100], dtype=np.float32)

        result = deltacamera.reproject_box_corners(box, cam_pinhole, cam_rotated)

        assert result[2] > 0  # width > 0
        assert result[3] > 0  # height > 0

    def test_reproject_box_side_midpoints(self, cam_pinhole, cam_rotated):
        """reproject_box_side_midpoints should give reasonable result."""
        box = np.array([100, 100, 100, 100], dtype=np.float32)

        result = deltacamera.reproject_box_side_midpoints(box, cam_pinhole, cam_rotated)

        assert result[2] > 0
        assert result[3] > 0

    def test_reproject_box_inscribed_ellipse(self, cam_pinhole, cam_rotated):
        """reproject_box_inscribed_ellipse should give reasonable result."""
        pytest.importorskip('boxlib', reason="boxlib.inscribed_ellipse_points not available")
        box = np.array([100, 100, 100, 100], dtype=np.float32)

        # Skip if boxlib doesn't have inscribed_ellipse_points
        import boxlib
        if not hasattr(boxlib, 'inscribed_ellipse_points'):
            pytest.skip("boxlib.inscribed_ellipse_points not available")

        result = deltacamera.reproject_box_inscribed_ellipse(box, cam_pinhole, cam_rotated)

        assert result[2] > 0
        assert result[3] > 0


# =============================================================================
# 3. reprojection.py - Mask warping
# =============================================================================


class TestReprojectMask:
    """Tests for reproject_mask and reproject_rle_mask - previously untested."""

    def test_reproject_mask_identity(self, cam_pinhole):
        """Reprojecting mask to same camera should preserve it."""
        # Create circular mask
        y, x = np.ogrid[:480, :640]
        mask = ((x - 320)**2 + (y - 240)**2 < 50**2).astype(np.uint8)

        mask_reproj = deltacamera.reproject_mask(
            mask, cam_pinhole, cam_pinhole, (480, 640)
        )

        # Should be nearly identical
        iou = (mask & mask_reproj).sum() / (mask | mask_reproj).sum()
        assert iou > 0.95

    def test_reproject_mask_preserves_area(self, cam_pinhole, cam_rotated):
        """Reprojecting mask with small rotation should preserve area roughly."""
        y, x = np.ogrid[:480, :640]
        mask = ((x - 320)**2 + (y - 240)**2 < 80**2).astype(np.uint8)

        mask_reproj = deltacamera.reproject_mask(
            mask, cam_pinhole, cam_rotated, (480, 640)
        )

        # Area should be roughly preserved (within 30%)
        area_orig = mask.sum()
        area_reproj = mask_reproj.sum()
        assert 0.7 < area_reproj / area_orig < 1.3

    def test_reproject_rle_mask_matches_dense(self, cam_pinhole, cam_rotated):
        """RLE mask reprojection should match dense mask reprojection."""
        from rlemasklib import RLEMask

        y, x = np.ogrid[:480, :640]
        mask = ((x - 320)**2 + (y - 240)**2 < 60**2).astype(np.uint8)

        # Dense reprojection
        mask_dense = deltacamera.reproject_mask(
            mask, cam_pinhole, cam_rotated, (480, 640)
        )

        # RLE reprojection
        rle = RLEMask.from_array(mask)
        rle_reproj = deltacamera.reproject_rle_mask(
            rle, cam_pinhole, cam_rotated, (480, 640)
        )
        mask_from_rle = rle_reproj.to_array()

        # Should be nearly identical (allow small boundary differences)
        intersection = (mask_dense & mask_from_rle).sum()
        union = (mask_dense | mask_from_rle).sum()
        iou = intersection / union if union > 0 else 1.0
        assert iou > 0.90


# =============================================================================
# 4. reprojection.py - Fast path
# =============================================================================


class TestReprojectImageFast:
    """Tests for reproject_image_fast - previously untested."""

    def test_fast_matches_regular_for_undistorted(self, cam_pinhole, cam_rotated):
        """Fast path should match regular path for undistorted cameras."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        img_regular = deltacamera.reproject_image(
            img, cam_pinhole, cam_rotated, (480, 640)
        )
        img_fast = deltacamera.reproject_image_fast(
            img, cam_pinhole, cam_rotated, (480, 640)
        )

        # Should be very similar (allow interpolation differences)
        # Compare only valid (non-zero) regions
        valid = (img_regular.sum(axis=-1) > 0) & (img_fast.sum(axis=-1) > 0)
        if valid.sum() > 0:
            diff = np.abs(
                img_regular[valid].astype(float) - img_fast[valid].astype(float)
            )
            assert np.percentile(diff, 95) < 10

    def test_fast_identity(self, cam_pinhole):
        """Fast reproject to same camera should be identity."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        img_reproj = deltacamera.reproject_image_fast(
            img, cam_pinhole, cam_pinhole, (480, 640)
        )

        # Center region should be identical
        center_orig = img[100:380, 100:540]
        center_reproj = img_reproj[100:380, 100:540]
        np.testing.assert_array_equal(center_orig, center_reproj)


# =============================================================================
# 5. sRGB color handling
# =============================================================================


class TestSRGB:
    """Tests for encode_srgb and decode_srgb - previously untested."""

    def test_srgb_roundtrip(self):
        """encode_srgb(decode_srgb(x)) should be near-identity."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        linear = deltacamera.decode_srgb(img)  # returns uint16
        back = deltacamera.encode_srgb(linear)  # takes uint16, returns uint8

        # Allow rounding errors of 1
        np.testing.assert_allclose(img, back, atol=1)

    def test_decode_srgb_returns_uint16(self):
        """Decoded sRGB should be uint16."""
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)

        linear = deltacamera.decode_srgb(img)

        assert linear.dtype == np.uint16

    def test_encode_srgb_returns_uint8(self):
        """Encoded sRGB should be uint8."""
        # Create uint16 linear image
        linear = np.array([[[0, 32768, 65535]]], dtype=np.uint16)

        encoded = deltacamera.encode_srgb(linear)

        assert encoded.dtype == np.uint8

    def test_srgb_extreme_values(self):
        """Test sRGB conversion at extreme values."""
        # Black and white
        img = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)

        linear = deltacamera.decode_srgb(img)
        back = deltacamera.encode_srgb(linear)

        # Allow rounding error of 1
        np.testing.assert_allclose(img, back, atol=1)


# =============================================================================
# 6. maps.py - Zero coverage currently
# =============================================================================


class TestMaps:
    """Tests for maps.py - previously zero coverage."""

    def test_make_maps_identity(self, cam_pinhole):
        """Maps for identical cameras should be approximately identity."""
        maps, valid = deltacamera.maps.make_maps_and_mask(
            cam_pinhole, cam_pinhole,
            input_imshape=(480, 640),
            output_imshape=(480, 640),
            precomp_undist_maps=None
        )
        # maps is shape (h, w, 2), maps[:,:,0] = x, maps[:,:,1] = y
        map_x, map_y = maps[:, :, 0], maps[:, :, 1]

        # Create expected identity maps
        y_coords, x_coords = np.mgrid[:480, :640].astype(np.float32)

        # Valid region should be large
        valid_mask = valid.to_array() if hasattr(valid, 'to_array') else valid
        assert valid_mask.sum() > 480 * 640 * 0.9

        # Where valid, maps should be identity
        np.testing.assert_allclose(map_x[valid_mask > 0], x_coords[valid_mask > 0], atol=1)
        np.testing.assert_allclose(map_y[valid_mask > 0], y_coords[valid_mask > 0], atol=1)

    def test_get_maps_cached_same_object(self, cam_pinhole):
        """Cached maps should return same object on repeated calls."""
        cam2 = cam_pinhole.zoom(1.1)

        result1 = deltacamera.maps.get_maps_and_mask_cached(
            cam_pinhole, cam2,
            input_imshape=(480, 640),
            output_imshape=(480, 640),
            precomp_undist_maps=None
        )
        result2 = deltacamera.maps.get_maps_and_mask_cached(
            cam_pinhole, cam2,
            input_imshape=(480, 640),
            output_imshape=(480, 640),
            precomp_undist_maps=None
        )

        # Should be exact same objects from cache
        assert result1[0] is result2[0]
        assert result1[1] is result2[1]

    def test_make_maps_with_distortion(self, cam_distorted, cam_pinhole):
        """Maps from distorted to undistorted camera should work."""
        maps, valid = deltacamera.maps.make_maps_and_mask(
            cam_distorted, cam_pinhole,
            input_imshape=(480, 640),
            output_imshape=(480, 640),
            precomp_undist_maps=None
        )
        # maps is shape (h, w, 2)
        map_x, map_y = maps[:, :, 0], maps[:, :, 1]

        assert map_x.shape == (480, 640)
        assert map_y.shape == (480, 640)
        # Should have some valid region
        valid_mask = valid.to_array() if hasattr(valid, 'to_array') else valid
        assert valid_mask.sum() > 0


# =============================================================================
# 7. Camera methods - undistort
# =============================================================================


class TestCameraUndistort:
    """Tests for Camera.undistort - previously untested."""

    def test_undistort_removes_distortion(self, cam_distorted):
        """camera.undistort() should produce camera with zero distortion."""
        cam_undist = cam_distorted.undistort(imshape=(480, 640))

        assert not cam_undist.has_distortion()

    def test_undistort_preserves_center(self, cam_distorted):
        """Undistorted camera should preserve image center approximately."""
        cam_undist = cam_distorted.undistort(imshape=(480, 640))

        # Center point should map to approximately same location
        center = np.array([[320, 240]], dtype=np.float32)
        center_reproj = deltacamera.reproject_image_points(
            center, cam_distorted, cam_undist
        )

        np.testing.assert_allclose(center_reproj, center, atol=5)

    def test_undistort_14param_with_tilt(self, cam_distorted_14param):
        """Undistort should work with 14-param model including tilt."""
        cam_undist = cam_distorted_14param.undistort(imshape=(480, 640))

        assert not cam_undist.has_distortion()

    def test_undistort_fisheye(self, cam_fisheye):
        """Undistort should work with fisheye camera."""
        cam_undist = cam_fisheye.undistort(imshape=(480, 640))

        assert not cam_undist.has_distortion()


# =============================================================================
# 8. Edge cases - distortion boundary behavior
# =============================================================================


class TestDistortionEdgeCases:
    """Tests for distortion edge cases."""

    def test_undistort_clip_vs_check(self):
        """clip_to_valid should give valid coords, check_validity should give NaN."""
        # Strong distortion with small valid region - must be padded to 14 elements
        d = np.zeros(14, dtype=np.float32)
        d[:8] = [0.5, -0.8, 0, 0, 0.3, 0.2, 0.1, 0.05]

        # Point far from center (likely outside valid region)
        points = np.array([[2.0, 2.0]], dtype=np.float32)

        result_clip = deltacamera.distortion.undistort_points(
            points, d, check_validity=False, clip_to_valid=True
        )
        result_check = deltacamera.distortion.undistort_points(
            points, d, check_validity=True, clip_to_valid=False
        )

        # Clipped should give valid (non-NaN) coords
        assert not np.isnan(result_clip).any()
        # check_validity should give NaN for out-of-bounds
        assert np.isnan(result_check).any()

    def test_distort_undistort_roundtrip(self, cam_distorted):
        """distort(undistort(x)) should be identity for valid points."""
        d = cam_distorted.get_distortion_coeffs(14)

        # Points near center (definitely valid)
        points = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.1, 0.05],
            [0.05, -0.1],
        ], dtype=np.float32)

        undistorted = deltacamera.distortion.undistort_points(
            points, d, check_validity=False
        )
        redistorted = deltacamera.distortion.distort_points(
            undistorted, d, check_validity=False
        )

        np.testing.assert_allclose(redistorted, points, atol=1e-5)

    def test_fisheye_wide_angle(self, cam_fisheye):
        """Fisheye should handle points at wide angles."""
        d = cam_fisheye.distortion_coeffs

        # Point at ~60° from optical axis (tan(60°) ≈ 1.73)
        points = np.array([[1.5, 0.0]], dtype=np.float32)

        result = deltacamera.distortion.distort_points_fisheye(
            points, d, check_validity=True
        )

        # Should give valid result (not NaN) for reasonable FOV
        assert not np.isnan(result).any()

    def test_14param_tilt_distort_undistort_roundtrip(self, cam_distorted_14param):
        """14-param with tilt should roundtrip correctly."""
        d = cam_distorted_14param.get_distortion_coeffs(14)

        points = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [-0.05, 0.08],
        ], dtype=np.float32)

        distorted = deltacamera.distortion.distort_points(
            points, d, check_validity=False
        )
        undistorted = deltacamera.distortion.undistort_points(
            distorted, d, check_validity=False
        )

        np.testing.assert_allclose(undistorted, points, atol=1e-4)
