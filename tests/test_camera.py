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
