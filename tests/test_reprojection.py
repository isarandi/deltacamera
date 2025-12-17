"""Tests for image and point reprojection."""

import numpy as np
import pytest

import lensform
import lensform.reprojection
from conftest import random_rotation_matrix, small_rotation_matrix


class TestPointReprojection:
    """Test point reprojection between cameras."""

    def test_identity_reprojection(self, simple_camera):
        """Reprojecting to same camera should be identity."""
        points = np.random.rand(50, 2).astype(np.float32) * [640, 480]

        reprojected = lensform.reprojection.reproject_image_points(
            points, simple_camera, simple_camera
        )

        np.testing.assert_allclose(reprojected, points, rtol=1e-5)

    def test_only_intrinsics_change(self):
        """Changing only intrinsics should be a linear transform."""
        camera1 = lensform.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )
        camera2 = lensform.Camera(
            intrinsic_matrix=[[600, 0, 400], [0, 600, 300], [0, 0, 1]],
        )

        # Principal point
        pp1 = np.array([[320, 240]], np.float32)
        reprojected = lensform.reprojection.reproject_image_points(pp1, camera1, camera2)

        # Principal point of camera1 (optical axis) should map to pp of camera2
        np.testing.assert_allclose(reprojected, [[400, 300]], rtol=1e-5)

    def test_rotation_moves_points(self):
        """Rotation should move off-center points."""
        camera1 = lensform.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )
        camera2 = camera1.copy()
        camera2.R = small_rotation_matrix(10) @ camera2.R

        # Point away from center
        point = np.array([[400, 300]], np.float32)
        reprojected = lensform.reprojection.reproject_image_points(point, camera1, camera2)

        # Should have moved
        assert not np.allclose(reprojected, point, atol=1)

    def test_reprojection_roundtrip(self):
        """Reprojecting A->B->A should recover original."""
        camera1 = lensform.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )
        camera2 = lensform.Camera(
            intrinsic_matrix=[[600, 0, 350], [0, 600, 280], [0, 0, 1]],
            rot_world_to_cam=small_rotation_matrix(5),
        )

        points = np.array([
            [320, 240],  # Center
            [200, 150],
            [450, 350],
        ], np.float32)

        intermediate = lensform.reprojection.reproject_image_points(points, camera1, camera2)
        recovered = lensform.reprojection.reproject_image_points(intermediate, camera2, camera1)

        np.testing.assert_allclose(recovered, points, rtol=1e-4, atol=0.1)


class TestImageReprojection:
    """Test image reprojection between cameras."""

    def test_identity_reprojection(self, simple_camera):
        """Reprojecting to same camera should preserve image."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        reprojected = lensform.reprojection.reproject_image(
            image, simple_camera, simple_camera, output_imshape=(480, 640),
            return_validity_mask=True
        )

        # reproject_image returns (image, mask) when return_validity_mask=True
        if isinstance(reprojected, tuple):
            reprojected, mask = reprojected
            valid = mask.to_array() if hasattr(mask, 'to_array') else mask
            if valid.any():
                np.testing.assert_array_equal(reprojected[valid], image[valid])
        else:
            # No mask returned, compare directly
            np.testing.assert_array_equal(reprojected, image)

    def test_output_shape(self, simple_camera):
        """Output should have correct shape."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        camera2 = simple_camera.copy()
        camera2.intrinsic_matrix[0, 2] = 400  # Different principal point

        reprojected = lensform.reprojection.reproject_image(
            image, simple_camera, camera2, output_imshape=(600, 800)
        )

        assert reprojected.shape == (600, 800, 3)

    def test_grayscale_image(self, simple_camera):
        """Should handle grayscale images."""
        image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        reprojected = lensform.reprojection.reproject_image(
            image, simple_camera, simple_camera, output_imshape=(480, 640)
        )

        assert reprojected.shape == (480, 640)


class TestReprojectionWithDistortion:
    """Test reprojection with distorted cameras."""

    def test_undistort_to_pinhole(self):
        """Reprojecting from distorted to undistorted camera."""
        distorted = lensform.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            distortion_coeffs=[-0.2, 0.1, 0, 0, 0],
        )
        pinhole = lensform.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )

        # Center point should be unchanged
        center = np.array([[320, 240]], np.float32)
        reprojected = lensform.reprojection.reproject_image_points(center, distorted, pinhole)

        np.testing.assert_allclose(reprojected, center, atol=0.1)

    def test_distort_from_pinhole(self):
        """Reprojecting from pinhole to distorted camera."""
        pinhole = lensform.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )
        distorted = lensform.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            distortion_coeffs=[-0.2, 0.1, 0, 0, 0],
        )

        # Center point should be unchanged
        center = np.array([[320, 240]], np.float32)
        reprojected = lensform.reprojection.reproject_image_points(center, pinhole, distorted)

        np.testing.assert_allclose(reprojected, center, atol=0.1)


class TestReprojectionEdgeCases:
    """Test edge cases in reprojection."""

    def test_empty_points(self, simple_camera):
        """Empty point arrays should not crash."""
        empty = np.zeros((0, 2), np.float32)

        result = lensform.reprojection.reproject_image_points(
            empty, simple_camera, simple_camera
        )

        assert result.shape == (0, 2)

    def test_single_point(self, simple_camera):
        """Single point should work."""
        point = np.array([[320, 240]], np.float32)

        result = lensform.reprojection.reproject_image_points(
            point, simple_camera, simple_camera
        )

        assert result.shape == (1, 2)
