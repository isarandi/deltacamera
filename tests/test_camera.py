"""Tests for Camera class coordinate transforms."""

import numpy as np
import pytest

import deltacamera
from conftest import random_rotation_matrix


class TestCameraWorldTransforms:
    """Test world <-> camera coordinate transforms."""

    def test_world_to_camera_roundtrip(self, positioned_camera):
        """world_to_camera and camera_to_world are inverses."""
        world_points = np.random.randn(100, 3).astype(np.float32) * 10

        camera_points = positioned_camera.world_to_camera(world_points)
        recovered = positioned_camera.camera_to_world(camera_points)

        np.testing.assert_allclose(recovered, world_points, rtol=1e-4, atol=1e-5)

    def test_optical_center_maps_to_origin(self, positioned_camera):
        """Optical center in world coords should map to origin in camera coords."""
        optical_center = positioned_camera.t.reshape(1, 3)
        camera_point = positioned_camera.world_to_camera(optical_center)

        np.testing.assert_allclose(camera_point, [[0, 0, 0]], atol=1e-6)

    def test_identity_camera(self):
        """Identity camera should not change world coordinates."""
        camera = deltacamera.Camera()
        world_points = np.random.randn(50, 3).astype(np.float32)

        camera_points = camera.world_to_camera(world_points)

        np.testing.assert_allclose(camera_points, world_points, atol=1e-6)


class TestCameraImageTransforms:
    """Test camera <-> image coordinate transforms."""

    def test_camera_to_image_no_distortion(self, simple_camera):
        """Camera to image without distortion."""
        # Point on optical axis at z=1
        cam_point = np.array([[0, 0, 1]], np.float32)
        img_point = simple_camera.camera_to_image(cam_point)

        # Should project to principal point
        pp = simple_camera.intrinsic_matrix[:2, 2]
        np.testing.assert_allclose(img_point, pp.reshape(1, 2), atol=1e-6)

    def test_camera_to_image_roundtrip_no_distortion(self, simple_camera):
        """Roundtrip without distortion."""
        # Points in front of camera
        cam_points = np.random.randn(100, 3).astype(np.float32)
        cam_points[:, 2] = np.abs(cam_points[:, 2]) + 0.5  # Ensure z > 0

        img_points = simple_camera.camera_to_image(cam_points)
        # image_to_camera returns 3D with specified depth (default 1)
        recovered_3d = simple_camera.image_to_camera(img_points)

        # Compare normalized coordinates (x/z, y/z)
        cam_normalized = cam_points[:, :2] / cam_points[:, 2:3]
        recovered_normalized = recovered_3d[:, :2] / recovered_3d[:, 2:3]
        # Use rtol=1e-4 for float32 precision
        np.testing.assert_allclose(recovered_normalized, cam_normalized, rtol=1e-4)

    def test_camera_to_image_roundtrip_with_distortion(self, distorted_camera):
        """Roundtrip with Brown-Conrady distortion in safe region."""
        # Generate points well within valid region (r < 0.25, away from boundary)
        cam_points = np.random.randn(100, 3).astype(np.float32)
        cam_points[:, 2] = np.abs(cam_points[:, 2]) + 1
        # Clip normalized radius to 0.25 (well below critical radius ~0.46)
        normalized = cam_points[:, :2] / cam_points[:, 2:3]
        radii = np.linalg.norm(normalized, axis=1, keepdims=True)
        scale = np.minimum(1.0, 0.25 / (radii + 1e-6))
        cam_points[:, :2] *= scale

        img_points = distorted_camera.camera_to_image(cam_points)
        valid_forward = ~np.isnan(img_points[:, 0])
        assert np.sum(valid_forward) == len(cam_points), "All points should be valid in safe region"

        recovered_3d = distorted_camera.image_to_camera(img_points)
        valid_backward = ~np.isnan(recovered_3d[:, 0])
        assert np.sum(valid_backward) == len(cam_points), "All recovered points should be valid"

        cam_normalized = cam_points[:, :2] / cam_points[:, 2:3]
        recovered_normalized = recovered_3d[:, :2] / recovered_3d[:, 2:3]
        np.testing.assert_allclose(recovered_normalized, cam_normalized, rtol=1e-4, atol=1e-5)

    def test_camera_to_image_roundtrip_near_boundary(self, distorted_camera):
        """Roundtrip near validity boundary has degraded but bounded precision."""
        # Generate points that may be near boundary (r up to ~0.45)
        cam_points = np.random.randn(100, 3).astype(np.float32) * 0.3
        cam_points[:, 2] = np.abs(cam_points[:, 2]) + 1

        img_points = distorted_camera.camera_to_image(cam_points)
        valid_forward = ~np.isnan(img_points[:, 0])

        if np.sum(valid_forward) > 0:
            recovered_3d = distorted_camera.image_to_camera(img_points[valid_forward])
            valid_backward = ~np.isnan(recovered_3d[:, 0])

            if np.sum(valid_backward) > 0:
                cam_normalized = cam_points[valid_forward][valid_backward, :2] / cam_points[valid_forward][valid_backward, 2:3]
                recovered_normalized = recovered_3d[valid_backward, :2] / recovered_3d[valid_backward, 2:3]
                # Relaxed tolerance for near-boundary points
                np.testing.assert_allclose(recovered_normalized, cam_normalized, rtol=1e-2, atol=1e-3)

    def test_camera_to_image_roundtrip_fisheye(self, fisheye_camera):
        """Roundtrip with fisheye distortion."""
        cam_points = np.random.randn(100, 3).astype(np.float32) * 0.3
        cam_points[:, 2] = np.abs(cam_points[:, 2]) + 1

        img_points = fisheye_camera.camera_to_image(cam_points)  # validation enabled
        valid_forward = ~np.isnan(img_points[:, 0])

        if np.sum(valid_forward) > 0:
            recovered_3d = fisheye_camera.image_to_camera(img_points[valid_forward])
            valid_backward = ~np.isnan(recovered_3d[:, 0])

            if np.sum(valid_backward) > 0:
                cam_normalized = cam_points[valid_forward][valid_backward, :2] / cam_points[valid_forward][valid_backward, 2:3]
                recovered_normalized = recovered_3d[valid_backward, :2] / recovered_3d[valid_backward, 2:3]
                np.testing.assert_allclose(recovered_normalized, cam_normalized, rtol=1e-4, atol=1e-5)


class TestCameraProperties:
    """Test camera property methods."""

    def test_has_distortion_false(self, simple_camera):
        """Camera without distortion coeffs should return False."""
        assert not simple_camera.has_distortion()

    def test_has_distortion_true(self, distorted_camera):
        """Camera with distortion coeffs should return True."""
        assert distorted_camera.has_distortion()

    def test_has_fisheye_distortion(self, fisheye_camera):
        """Fisheye camera should be identified correctly."""
        assert fisheye_camera.has_fisheye_distortion()
        assert not fisheye_camera.has_nonfisheye_distortion()

    def test_has_nonfisheye_distortion(self, distorted_camera):
        """Brown-Conrady camera should be identified correctly."""
        assert distorted_camera.has_nonfisheye_distortion()
        assert not distorted_camera.has_fisheye_distortion()


class TestCameraOperations:
    """Test camera modification operations."""

    def test_zoom(self, simple_camera):
        """Zoom should scale focal length."""
        original_fx = simple_camera.intrinsic_matrix[0, 0]
        original_fy = simple_camera.intrinsic_matrix[1, 1]

        zoomed = simple_camera.copy()
        zoomed.zoom(2.0)

        np.testing.assert_allclose(zoomed.intrinsic_matrix[0, 0], original_fx * 2)
        np.testing.assert_allclose(zoomed.intrinsic_matrix[1, 1], original_fy * 2)

    def test_scale_output(self, simple_camera):
        """Scale output should scale entire intrinsic matrix."""
        original = simple_camera.intrinsic_matrix.copy()

        scaled = simple_camera.copy()
        scaled.scale_output(0.5)

        np.testing.assert_allclose(scaled.intrinsic_matrix[:2], original[:2] * 0.5)

    def test_copy_independence(self, simple_camera):
        """Copied camera should be independent."""
        copy = simple_camera.copy()
        copy.zoom(2.0)

        # Original should be unchanged
        assert simple_camera.intrinsic_matrix[0, 0] != copy.intrinsic_matrix[0, 0]

    def test_rotate(self, simple_camera):
        """Rotate should change the rotation matrix."""
        original_R = simple_camera.R.copy()

        rotated = simple_camera.copy()
        rotated.rotate(yaw=0.1)

        # R should have changed
        assert not np.allclose(rotated.R, original_R)


class TestCameraEdgeCases:
    """Test edge cases."""

    def test_single_point(self, simple_camera):
        """Single point should work."""
        point = np.array([[0, 0, 1]], np.float32)
        result = simple_camera.camera_to_image(point)
        assert result.shape == (1, 2)

    def test_points_behind_camera(self, simple_camera):
        """Points with z <= 0 should produce inf or handle gracefully."""
        behind = np.array([[0, 0, -1], [1, 1, 0]], np.float32)
        result = simple_camera.camera_to_image(behind)
        # Either inf, nan, or very large values are acceptable
        assert np.all(~np.isfinite(result) | (np.abs(result) > 1e6))


class TestImageTransforms:
    """Test image transformation methods with distorted cameras."""

    @pytest.fixture
    def jrdb_camera(self):
        """Create camera from JRDB calibration (Brown-Conrady 5-param)."""
        data = np.load("tests/data/jrdb_calibrations.npz")
        return deltacamera.Camera(
            intrinsic_matrix=data["intrinsic_matrices"][0],
            distortion_coeffs=data["distortion_coeffs"][0],
            image_shape=(data["resolutions"][0, 1], data["resolutions"][0, 0]),
        )

    @pytest.fixture
    def egohumans_camera(self):
        """Create camera from EgoHumans calibration (fisheye 4-param)."""
        data = np.load("tests/data/egohumans_fisheye_calibrations.npz")
        return deltacamera.Camera(
            intrinsic_matrix=data["intrinsic_matrices"][0],
            distortion_coeffs=data["distortion_coeffs"][0],
            image_shape=(data["resolutions"][0, 1], data["resolutions"][0, 0]),
        )

    def _generate_world_points(self, n=50):
        """Generate random world points in front of camera."""
        points = np.random.randn(n, 3).astype(np.float32) * 2
        points[:, 2] = np.abs(points[:, 2]) + 3  # Ensure z > 0
        return points

    def _filter_valid_points(self, cam, world_points):
        """Project points and keep only those with valid image coordinates."""
        img_pts = cam.world_to_image(world_points)
        valid = (
            np.isfinite(img_pts).all(axis=1) &
            (img_pts[:, 0] >= 0) & (img_pts[:, 0] < cam.image_shape[1]) &
            (img_pts[:, 1] >= 0) & (img_pts[:, 1] < cam.image_shape[0])
        )
        return world_points[valid], img_pts[valid]

    # -------------------------------------------------------------------------
    # Horizontal flip tests
    # -------------------------------------------------------------------------

    def test_hflip_jrdb(self, jrdb_camera):
        """Test image_hflipped with JRDB Brown-Conrady camera."""
        world_pts, orig_img_pts = self._filter_valid_points(
            jrdb_camera, self._generate_world_points(100)
        )
        assert len(world_pts) > 10, "Need enough valid points"

        flipped_cam = jrdb_camera.image_hflipped()
        flipped_img_pts = flipped_cam.world_to_image(world_pts)

        # After hflip: x' = width - 1 - x, y' = y
        width = jrdb_camera.image_shape[1]
        expected_x = width - 1 - orig_img_pts[:, 0]

        np.testing.assert_allclose(flipped_img_pts[:, 0], expected_x, rtol=1e-4, atol=0.5)
        np.testing.assert_allclose(flipped_img_pts[:, 1], orig_img_pts[:, 1], rtol=1e-4, atol=0.5)

    def test_hflip_egohumans(self, egohumans_camera):
        """Test image_hflipped with EgoHumans fisheye camera."""
        world_pts, orig_img_pts = self._filter_valid_points(
            egohumans_camera, self._generate_world_points(100)
        )
        assert len(world_pts) > 10, "Need enough valid points"

        flipped_cam = egohumans_camera.image_hflipped()
        flipped_img_pts = flipped_cam.world_to_image(world_pts)

        # After hflip: x' = width - 1 - x, y' = y
        width = egohumans_camera.image_shape[1]
        expected_x = width - 1 - orig_img_pts[:, 0]

        np.testing.assert_allclose(flipped_img_pts[:, 0], expected_x, rtol=1e-4, atol=0.5)
        np.testing.assert_allclose(flipped_img_pts[:, 1], orig_img_pts[:, 1], rtol=1e-4, atol=0.5)

    # -------------------------------------------------------------------------
    # Rot90 tests
    # -------------------------------------------------------------------------

    def test_rot90_jrdb(self, jrdb_camera):
        """Test image_rot90 with JRDB Brown-Conrady camera."""
        world_pts, orig_img_pts = self._filter_valid_points(
            jrdb_camera, self._generate_world_points(100)
        )
        assert len(world_pts) > 10, "Need enough valid points"

        rotated_cam = jrdb_camera.image_rot90(k=1)
        rotated_img_pts = rotated_cam.world_to_image(world_pts)

        # After rot90 k=1: (x, y) -> (H - 1 - y, x) where H is original height
        height = jrdb_camera.image_shape[0]
        expected_x = height - 1 - orig_img_pts[:, 1]
        expected_y = orig_img_pts[:, 0]

        np.testing.assert_allclose(rotated_img_pts[:, 0], expected_x, rtol=1e-4, atol=0.5)
        np.testing.assert_allclose(rotated_img_pts[:, 1], expected_y, rtol=1e-4, atol=0.5)

    def test_rot90_egohumans(self, egohumans_camera):
        """Test image_rot90 with EgoHumans fisheye camera."""
        world_pts, orig_img_pts = self._filter_valid_points(
            egohumans_camera, self._generate_world_points(100)
        )
        assert len(world_pts) > 10, "Need enough valid points"

        rotated_cam = egohumans_camera.image_rot90(k=1)
        rotated_img_pts = rotated_cam.world_to_image(world_pts)

        # After rot90 k=1: (x, y) -> (H - 1 - y, x) where H is original height
        height = egohumans_camera.image_shape[0]
        expected_x = height - 1 - orig_img_pts[:, 1]
        expected_y = orig_img_pts[:, 0]

        np.testing.assert_allclose(rotated_img_pts[:, 0], expected_x, rtol=1e-4, atol=0.5)
        np.testing.assert_allclose(rotated_img_pts[:, 1], expected_y, rtol=1e-4, atol=0.5)

    def test_rot90_roundtrip(self, jrdb_camera):
        """Test that 4 rot90 operations return to original."""
        world_pts, orig_img_pts = self._filter_valid_points(
            jrdb_camera, self._generate_world_points(50)
        )
        assert len(world_pts) > 5

        cam4 = jrdb_camera.image_rot90(k=4)
        img_pts_4 = cam4.world_to_image(world_pts)

        np.testing.assert_allclose(img_pts_4, orig_img_pts, rtol=1e-4, atol=0.5)

    # -------------------------------------------------------------------------
    # Rotated (arbitrary angle) tests
    # -------------------------------------------------------------------------

    def test_rotated_jrdb(self, jrdb_camera):
        """Test image_rotated with JRDB Brown-Conrady camera."""
        world_pts, orig_img_pts = self._filter_valid_points(
            jrdb_camera, self._generate_world_points(100)
        )
        assert len(world_pts) > 10, "Need enough valid points"

        angle = np.pi / 6  # 30 degrees
        rotated_cam = jrdb_camera.image_rotated(angle)
        rotated_img_pts = rotated_cam.world_to_image(world_pts)

        # Compute expected coords by rotating around image center
        h, w = jrdb_camera.image_shape
        center = np.array([(w - 1) / 2, (h - 1) / 2], dtype=np.float32)
        cos, sin = np.cos(angle), np.sin(angle)
        R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)

        centered = orig_img_pts - center
        expected = (R @ centered.T).T + center

        np.testing.assert_allclose(rotated_img_pts, expected, rtol=1e-4, atol=0.5)

    def test_rotated_egohumans(self, egohumans_camera):
        """Test image_rotated with EgoHumans fisheye camera."""
        world_pts, orig_img_pts = self._filter_valid_points(
            egohumans_camera, self._generate_world_points(100)
        )
        assert len(world_pts) > 10, "Need enough valid points"

        angle = -np.pi / 4  # -45 degrees
        rotated_cam = egohumans_camera.image_rotated(angle)
        rotated_img_pts = rotated_cam.world_to_image(world_pts)

        # Compute expected coords by rotating around image center
        h, w = egohumans_camera.image_shape
        center = np.array([(w - 1) / 2, (h - 1) / 2], dtype=np.float32)
        cos, sin = np.cos(angle), np.sin(angle)
        R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)

        centered = orig_img_pts - center
        expected = (R @ centered.T).T + center

        np.testing.assert_allclose(rotated_img_pts, expected, rtol=1e-4, atol=0.5)

    def test_rotated_roundtrip(self, jrdb_camera):
        """Test that rotating by 2*pi returns to original."""
        world_pts, orig_img_pts = self._filter_valid_points(
            jrdb_camera, self._generate_world_points(50)
        )
        assert len(world_pts) > 5

        rotated_cam = jrdb_camera.image_rotated(2 * np.pi)
        rotated_img_pts = rotated_cam.world_to_image(world_pts)

        np.testing.assert_allclose(rotated_img_pts, orig_img_pts, rtol=1e-4, atol=0.5)

    # -------------------------------------------------------------------------
    # Image reprojection tests (pixel-level verification)
    # -------------------------------------------------------------------------

    @pytest.fixture
    def weak_distortion_14param(self):
        """Camera with weak 14-param distortion for image reprojection tests.

        Uses weaker distortion than real cameras to ensure the valid region
        covers most of the image after transformations like hflip and rot90.
        """
        d14 = np.zeros(14, dtype=np.float32)
        np.random.seed(42)
        d14[0] = -0.05    # k1 (weak radial)
        d14[1] = 0.01     # k2
        d14[2:4] = np.random.randn(2) * 0.0001  # p1, p2 (weak tangential)
        d14[4] = 0.001    # k3
        d14[5:8] = np.random.randn(3) * 0.0001  # k4, k5, k6
        d14[8:12] = np.random.randn(4) * 0.00005  # s1, s2, s3, s4
        d14[12:14] = np.random.randn(2) * 0.005   # tau_x, tau_y (weak tilt)
        return deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            distortion_coeffs=d14,
            image_shape=(480, 640),
        )

    @pytest.fixture
    def generic_camera(self):
        """Camera with skew, non-equal focal lengths, and 14-param distortion.

        This is the most general Brown-Conrady camera configuration to test
        that all transformation code handles the full parameter space.
        """
        d14 = np.zeros(14, dtype=np.float32)
        np.random.seed(123)
        d14[0] = -0.08    # k1
        d14[1] = 0.02     # k2
        d14[2:4] = np.random.randn(2) * 0.0002  # p1, p2
        d14[4] = 0.005    # k3
        d14[5:8] = np.random.randn(3) * 0.0002  # k4, k5, k6
        d14[8:12] = np.random.randn(4) * 0.0001  # s1, s2, s3, s4
        d14[12:14] = np.random.randn(2) * 0.02   # tau_x, tau_y
        return deltacamera.Camera(
            intrinsic_matrix=[[520, 2.5, 325], [0, 480, 242], [0, 0, 1]],
            distortion_coeffs=d14,
            image_shape=(480, 640),
        )

    def _get_test_image(self, shape):
        """Get astronaut image resized to target shape."""
        from skimage.data import astronaut
        from skimage.transform import resize
        img = astronaut()
        return (resize(img, shape[:2], anti_aliasing=True) * 255).astype(np.uint8)

    def _scaled_camera(self, cam, scale):
        """Return a scaled copy of the camera."""
        c = cam.copy()
        c.scale_output(scale)
        return c

    def test_hflip_image_14param(self, weak_distortion_14param):
        """Test image_hflipped reprojection matches np.fliplr (14-param)."""
        from deltacamera import reproject_image

        cam = weak_distortion_14param
        img = self._get_test_image(cam.image_shape)

        flipped_cam = cam.image_hflipped()
        reprojected = reproject_image(img, cam, flipped_cam)
        expected = np.fliplr(img)

        margin = 10
        diff = np.abs(reprojected[margin:-margin, margin:-margin].astype(float) -
                      expected[margin:-margin, margin:-margin].astype(float))
        assert np.mean(diff) < 2.0, f"Mean pixel diff {np.mean(diff):.2f} > 2.0"

    def test_hflip_image_fisheye(self, egohumans_camera):
        """Test image_hflipped reprojection matches np.fliplr (fisheye)."""
        from deltacamera import reproject_image

        cam = self._scaled_camera(egohumans_camera, 0.1)
        img = self._get_test_image(cam.image_shape)

        flipped_cam = cam.image_hflipped()
        reprojected = reproject_image(img, cam, flipped_cam)
        expected = np.fliplr(img)

        margin = 10
        diff = np.abs(reprojected[margin:-margin, margin:-margin].astype(float) -
                      expected[margin:-margin, margin:-margin].astype(float))
        # Higher threshold for fisheye due to stronger distortion causing more interpolation error
        assert np.mean(diff) < 5.0, f"Mean pixel diff {np.mean(diff):.2f} > 5.0"

    def test_rot90_image_14param(self, weak_distortion_14param):
        """Test image_rot90 reprojection matches np.rot90 (14-param)."""
        from deltacamera import reproject_image

        cam = weak_distortion_14param
        img = self._get_test_image(cam.image_shape)

        rotated_cam = cam.image_rot90(k=1)
        reprojected = reproject_image(img, cam, rotated_cam)
        # image_rot90(k=1) does CW rotation, np.rot90(k=-1) is also CW
        expected = np.rot90(img, k=-1)

        margin = 10
        diff = np.abs(reprojected[margin:-margin, margin:-margin].astype(float) -
                      expected[margin:-margin, margin:-margin].astype(float))
        assert np.mean(diff) < 2.0, f"Mean pixel diff {np.mean(diff):.2f} > 2.0"

    def test_rot90_image_fisheye(self, egohumans_camera):
        """Test image_rot90 reprojection matches np.rot90 (fisheye)."""
        from deltacamera import reproject_image

        cam = self._scaled_camera(egohumans_camera, 0.1)
        img = self._get_test_image(cam.image_shape)

        rotated_cam = cam.image_rot90(k=1)
        reprojected = reproject_image(img, cam, rotated_cam)
        expected = np.rot90(img, k=-1)

        margin = 10
        diff = np.abs(reprojected[margin:-margin, margin:-margin].astype(float) -
                      expected[margin:-margin, margin:-margin].astype(float))
        # Higher threshold for fisheye due to stronger distortion causing more interpolation error
        assert np.mean(diff) < 5.0, f"Mean pixel diff {np.mean(diff):.2f} > 5.0"

    # -------------------------------------------------------------------------
    # Generic camera tests (skew + non-equal focals + 14-param distortion)
    # -------------------------------------------------------------------------

    def test_hflip_generic(self, generic_camera):
        """Test image_hflipped with generic camera (skew, fx != fy, 14-param)."""
        world_pts, orig_img_pts = self._filter_valid_points(
            generic_camera, self._generate_world_points(100)
        )
        assert len(world_pts) > 10, "Need enough valid points"

        flipped_cam = generic_camera.image_hflipped()
        flipped_img_pts = flipped_cam.world_to_image(world_pts)

        width = generic_camera.image_shape[1]
        expected_x = width - 1 - orig_img_pts[:, 0]

        np.testing.assert_allclose(flipped_img_pts[:, 0], expected_x, rtol=1e-4, atol=0.5)
        np.testing.assert_allclose(flipped_img_pts[:, 1], orig_img_pts[:, 1], rtol=1e-4, atol=0.5)

    def test_rot90_generic(self, generic_camera):
        """Test image_rot90 with generic camera (skew, fx != fy, 14-param)."""
        world_pts, orig_img_pts = self._filter_valid_points(
            generic_camera, self._generate_world_points(100)
        )
        assert len(world_pts) > 10, "Need enough valid points"

        rotated_cam = generic_camera.image_rot90(k=1)
        rotated_img_pts = rotated_cam.world_to_image(world_pts)

        height = generic_camera.image_shape[0]
        expected_x = height - 1 - orig_img_pts[:, 1]
        expected_y = orig_img_pts[:, 0]

        np.testing.assert_allclose(rotated_img_pts[:, 0], expected_x, rtol=1e-4, atol=0.5)
        np.testing.assert_allclose(rotated_img_pts[:, 1], expected_y, rtol=1e-4, atol=0.5)

    def test_rotated_generic(self, generic_camera):
        """Test image_rotated with generic camera (skew, fx != fy, 14-param)."""
        world_pts, orig_img_pts = self._filter_valid_points(
            generic_camera, self._generate_world_points(100)
        )
        assert len(world_pts) > 10, "Need enough valid points"

        angle = np.pi / 5  # 36 degrees (non-90-multiple to test skew handling)
        rotated_cam = generic_camera.image_rotated(angle)
        rotated_img_pts = rotated_cam.world_to_image(world_pts)

        h, w = generic_camera.image_shape
        center = np.array([(w - 1) / 2, (h - 1) / 2], dtype=np.float32)
        cos, sin = np.cos(angle), np.sin(angle)
        R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)

        centered = orig_img_pts - center
        expected = (R @ centered.T).T + center

        np.testing.assert_allclose(rotated_img_pts, expected, rtol=1e-4, atol=0.5)

    def test_hflip_image_generic(self, generic_camera):
        """Test image_hflipped reprojection matches np.fliplr (generic camera)."""
        from deltacamera import reproject_image

        cam = generic_camera
        img = self._get_test_image(cam.image_shape)

        flipped_cam = cam.image_hflipped()
        reprojected = reproject_image(img, cam, flipped_cam)
        expected = np.fliplr(img)

        margin = 15
        diff = np.abs(reprojected[margin:-margin, margin:-margin].astype(float) -
                      expected[margin:-margin, margin:-margin].astype(float))
        assert np.mean(diff) < 3.0, f"Mean pixel diff {np.mean(diff):.2f} > 3.0"

    def test_rot90_image_generic(self, generic_camera):
        """Test image_rot90 reprojection matches np.rot90 (generic camera)."""
        from deltacamera import reproject_image

        cam = generic_camera
        img = self._get_test_image(cam.image_shape)

        rotated_cam = cam.image_rot90(k=1)
        reprojected = reproject_image(img, cam, rotated_cam)
        expected = np.rot90(img, k=-1)

        margin = 15
        diff = np.abs(reprojected[margin:-margin, margin:-margin].astype(float) -
                      expected[margin:-margin, margin:-margin].astype(float))
        assert np.mean(diff) < 3.0, f"Mean pixel diff {np.mean(diff):.2f} > 3.0"
