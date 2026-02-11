"""Tests for image and point reprojection."""

import numpy as np
import pytest

import deltacamera
import deltacamera.reprojection
from conftest import random_rotation_matrix, small_rotation_matrix


class TestPointReprojection:
    """Test point reprojection between cameras."""

    def test_identity_reprojection(self, simple_camera):
        """Reprojecting to same camera should be identity."""
        points = np.random.rand(50, 2).astype(np.float32) * [640, 480]

        reprojected = deltacamera.reprojection.reproject_image_points(
            points, simple_camera, simple_camera
        )

        np.testing.assert_allclose(reprojected, points, rtol=1e-5)

    def test_only_intrinsics_change(self):
        """Changing only intrinsics should be a linear transform."""
        camera1 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )
        camera2 = deltacamera.Camera(
            intrinsic_matrix=[[600, 0, 400], [0, 600, 300], [0, 0, 1]],
        )

        # Principal point
        pp1 = np.array([[320, 240]], np.float32)
        reprojected = deltacamera.reprojection.reproject_image_points(pp1, camera1, camera2)

        # Principal point of camera1 (optical axis) should map to pp of camera2
        np.testing.assert_allclose(reprojected, [[400, 300]], rtol=1e-5)

    def test_rotation_moves_points(self):
        """Rotation should move off-center points."""
        camera1 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )
        camera2 = camera1.copy()
        camera2.R = small_rotation_matrix(10) @ camera2.R

        # Point away from center
        point = np.array([[400, 300]], np.float32)
        reprojected = deltacamera.reprojection.reproject_image_points(point, camera1, camera2)

        # Should have moved
        assert not np.allclose(reprojected, point, atol=1)

    def test_reprojection_roundtrip(self):
        """Reprojecting A->B->A should recover original."""
        camera1 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )
        camera2 = deltacamera.Camera(
            intrinsic_matrix=[[600, 0, 350], [0, 600, 280], [0, 0, 1]],
            rot_world_to_cam=small_rotation_matrix(5),
        )

        points = np.array([
            [320, 240],  # Center
            [200, 150],
            [450, 350],
        ], np.float32)

        intermediate = deltacamera.reprojection.reproject_image_points(points, camera1, camera2)
        recovered = deltacamera.reprojection.reproject_image_points(intermediate, camera2, camera1)

        np.testing.assert_allclose(recovered, points, rtol=1e-4, atol=0.1)


class TestImageReprojection:
    """Test image reprojection between cameras."""

    def test_identity_reprojection(self, simple_camera):
        """Reprojecting to same camera should preserve image."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        reprojected = deltacamera.reprojection.reproject_image(
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

        # Use immutable API: shift principal point by (80, 0) to move from 320 to 400
        camera2 = simple_camera.image_shifted([80, 0])

        reprojected = deltacamera.reprojection.reproject_image(
            image, simple_camera, camera2, output_imshape=(600, 800)
        )

        assert reprojected.shape == (600, 800, 3)

    def test_grayscale_image(self, simple_camera):
        """Should handle grayscale images."""
        image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        reprojected = deltacamera.reprojection.reproject_image(
            image, simple_camera, simple_camera, output_imshape=(480, 640)
        )

        assert reprojected.shape == (480, 640)


class TestReprojectionWithDistortion:
    """Test reprojection with distorted cameras."""

    def test_undistort_to_pinhole(self):
        """Reprojecting from distorted to undistorted camera."""
        distorted = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            distortion_coeffs=[-0.2, 0.1, 0, 0, 0],
        )
        pinhole = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )

        # Center point should be unchanged
        center = np.array([[320, 240]], np.float32)
        reprojected = deltacamera.reprojection.reproject_image_points(center, distorted, pinhole)

        np.testing.assert_allclose(reprojected, center, atol=0.1)

    def test_distort_from_pinhole(self):
        """Reprojecting from pinhole to distorted camera."""
        pinhole = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        )
        distorted = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            distortion_coeffs=[-0.2, 0.1, 0, 0, 0],
        )

        # Center point should be unchanged
        center = np.array([[320, 240]], np.float32)
        reprojected = deltacamera.reprojection.reproject_image_points(center, pinhole, distorted)

        np.testing.assert_allclose(reprojected, center, atol=0.1)


class TestDepthMapReprojection:
    """Test depth map reprojection between cameras."""

    def test_identity(self):
        """Reprojecting to same camera should leave depth unchanged."""
        camera = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            image_shape=(480, 640),
        )
        depth = np.random.rand(480, 640).astype(np.float32) * 10 + 1

        result = deltacamera.reprojection.reproject_depth_map(depth, camera, camera)

        valid = ~np.isnan(result)
        np.testing.assert_allclose(result[valid], depth[valid], rtol=1e-5)

    def test_only_intrinsics_change(self):
        """Changing only intrinsics (same R) should not change depth values."""
        camera1 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            image_shape=(480, 640),
        )
        camera2 = deltacamera.Camera(
            intrinsic_matrix=[[600, 0, 400], [0, 600, 300], [0, 0, 1]],
            image_shape=(480, 640),
        )

        depth = np.full((480, 640), 5.0, dtype=np.float32)

        result = deltacamera.reprojection.reproject_depth_map(depth, camera1, camera2)

        # Where valid, depth should be 5.0 (z-factor is 1.0 when R is the same)
        valid = ~np.isnan(result)
        assert valid.any()
        np.testing.assert_allclose(result[valid], 5.0, rtol=1e-5)

    def test_rotation_analytical(self):
        """Rotation around Y-axis: depth at principal point = D / cos(theta)."""
        from scipy.spatial.transform import Rotation

        theta_deg = 15
        theta_rad = np.deg2rad(theta_deg)

        camera1 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            image_shape=(480, 640),
        )
        R_rot = Rotation.from_rotvec([0, theta_rad, 0]).as_matrix().astype(np.float32)
        camera2 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            rot_world_to_cam=R_rot,
            image_shape=(480, 640),
        )

        D = 10.0
        depth = np.full((480, 640), D, dtype=np.float32)

        result = deltacamera.reprojection.reproject_depth_map(depth, camera1, camera2)

        # At the principal point of camera2 (320, 240), the ray is [0, 0, 1] in camera2,
        # rotated to camera1 frame: z-component = cos(theta)
        # So Z_new = D / cos(theta)
        expected = D / np.cos(theta_rad)
        np.testing.assert_allclose(result[240, 320], expected, rtol=1e-4)

    def test_roundtrip(self):
        """A->B->A should recover original depth."""
        from scipy.spatial.transform import Rotation

        camera1 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            image_shape=(480, 640),
        )
        R_rot = Rotation.from_rotvec([0.05, 0.03, -0.02]).as_matrix().astype(np.float32)
        camera2 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            rot_world_to_cam=R_rot,
            image_shape=(480, 640),
        )

        depth = np.full((480, 640), 5.0, dtype=np.float32)

        intermediate = deltacamera.reprojection.reproject_depth_map(depth, camera1, camera2)
        recovered = deltacamera.reprojection.reproject_depth_map(intermediate, camera2, camera1)

        # Center region should survive the round-trip
        center = recovered[180:300, 200:440]
        valid = ~np.isnan(center)
        assert valid.sum() > 100
        np.testing.assert_allclose(center[valid], 5.0, rtol=1e-2)

    def test_with_distortion_same_rotation(self):
        """Distorted->pinhole with same R: depth should be unchanged."""
        distorted = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            distortion_coeffs=[-0.2, 0.1, 0, 0, 0],
            image_shape=(480, 640),
        )
        pinhole = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            image_shape=(480, 640),
        )

        depth = np.full((480, 640), 7.0, dtype=np.float32)

        result = deltacamera.reprojection.reproject_depth_map(depth, distorted, pinhole)

        # Same R, so z-factor is 1.0; depth should be preserved where valid
        valid = ~np.isnan(result)
        assert valid.any()
        np.testing.assert_allclose(result[valid], 7.0, rtol=1e-4)

    def test_with_distortion_and_rotation(self):
        """Distorted new camera with rotation: z-factor uses undistorted rays."""
        from scipy.spatial.transform import Rotation

        theta_rad = np.deg2rad(10)
        R_rot = Rotation.from_rotvec([0, theta_rad, 0]).as_matrix().astype(np.float32)

        camera1 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            image_shape=(480, 640),
        )
        camera2 = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            distortion_coeffs=[-0.2, 0.1, 0, 0, 0],
            rot_world_to_cam=R_rot,
            image_shape=(480, 640),
        )

        D = 10.0
        depth = np.full((480, 640), D, dtype=np.float32)

        result = deltacamera.reprojection.reproject_depth_map(depth, camera1, camera2)

        # At principal point of camera2 (distortion is zero there),
        # undistorted ray is [0, 0, 1], rotated z-component = cos(theta)
        expected = D / np.cos(theta_rad)
        np.testing.assert_allclose(result[240, 320], expected, rtol=1e-3)

    def test_ground_truth_pyrender(self):
        """Ground truth: render depth from two rotations, distort both, reproject, compare."""
        import os
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        import pyrender
        import trimesh
        from scipy.spatial.transform import Rotation

        h, w = 480, 640
        fx, fy, cx, cy = 500, 500, 320, 240
        dist_coeffs1 = [-0.6, 0.3, 0, 0, 0]
        dist_coeffs2 = [-0.5, 0.25, 0.002, -0.002, 0.1]

        # Two rotations (same optical center)
        R1 = np.eye(3, dtype=np.float32)
        R2 = Rotation.from_rotvec([0.0, np.deg2rad(20), 0.0]).as_matrix().astype(np.float32)

        # Render depth from both viewpoints using pyrender
        scene = pyrender.Scene()
        box = trimesh.creation.box(extents=[2, 2, 0.5])
        box.apply_translation([0, 0, 5])
        scene.add(pyrender.Mesh.from_trimesh(box))

        floor = trimesh.creation.box(extents=[10, 10, 0.1])
        floor.apply_translation([0, 3, 8])
        scene.add(pyrender.Mesh.from_trimesh(floor))

        renderer = pyrender.OffscreenRenderer(w, h)

        def render_depth(R):
            # pyrender uses OpenGL convention (x-right, y-up, z-backward)
            # deltacamera uses CV convention (x-right, y-down, z-forward)
            # Convert: flip y and z axes
            R_gl = R.copy()
            R_gl[1] = -R_gl[1]
            R_gl[2] = -R_gl[2]
            pose = np.eye(4)
            pose[:3, :3] = R_gl.T  # pyrender expects cam-to-world
            cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.1, zfar=100)
            cam_node = scene.add(cam, pose=pose)
            _, depth = renderer.render(scene)
            scene.remove_node(cam_node)
            return depth.astype(np.float32)

        depth1 = render_depth(R1)
        depth2 = render_depth(R2)
        renderer.delete()

        # Pinhole cameras (what pyrender rendered)
        pinhole1 = deltacamera.Camera(
            intrinsic_matrix=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            rot_world_to_cam=R1, image_shape=(h, w),
        )
        pinhole2 = deltacamera.Camera(
            intrinsic_matrix=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            rot_world_to_cam=R2, image_shape=(h, w),
        )

        # Distorted cameras (same intrinsics + rotation, add distortion)
        dist1 = deltacamera.Camera(
            intrinsic_matrix=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            distortion_coeffs=dist_coeffs1, rot_world_to_cam=R1, image_shape=(h, w),
        )
        dist2 = deltacamera.Camera(
            intrinsic_matrix=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            distortion_coeffs=dist_coeffs2, rot_world_to_cam=R2, image_shape=(h, w),
        )

        # Warp rendered depth to distorted cameras
        depth1_dist = deltacamera.reproject_depth_map(depth1, pinhole1, dist1)
        depth2_dist = deltacamera.reproject_depth_map(depth2, pinhole2, dist2)

        # Reproject distorted depth from cam1 to cam2
        result = deltacamera.reproject_depth_map(depth1_dist, dist1, dist2)

        # Compare with ground truth. INTER_NEAREST causes ±0.5px errors per
        # reprojection, which at depth discontinuities yields large outliers.
        # We verify that 99.5%+ of valid pixels match closely.
        valid = ~np.isnan(result) & ~np.isnan(depth2_dist) & (depth2_dist > 0)
        assert valid.sum() > 1000
        abs_err = np.abs(result[valid] - depth2_dist[valid])
        pct_good = np.mean(abs_err < 0.05)
        assert pct_good > 0.99, f"Only {pct_good:.1%} of pixels within tolerance"
        assert np.percentile(abs_err, 95) < 0.005

    def test_return_validity_mask(self):
        """return_validity_mask=True should return a tuple."""
        camera = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            image_shape=(480, 640),
        )
        depth = np.full((480, 640), 5.0, dtype=np.float32)

        result = deltacamera.reprojection.reproject_depth_map(
            depth, camera, camera, return_validity_mask=True
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        depth_out, mask = result
        assert depth_out.shape == (480, 640)


class TestReprojectionEdgeCases:
    """Test edge cases in reprojection."""

    def test_empty_points(self, simple_camera):
        """Empty point arrays should not crash."""
        empty = np.zeros((0, 2), np.float32)

        result = deltacamera.reprojection.reproject_image_points(
            empty, simple_camera, simple_camera
        )

        assert result.shape == (0, 2)

    def test_single_point(self, simple_camera):
        """Single point should work."""
        point = np.array([[320, 240]], np.float32)

        result = deltacamera.reprojection.reproject_image_points(
            point, simple_camera, simple_camera
        )

        assert result.shape == (1, 2)
