"""Tests for Camera.rotate_image distortion coefficient transformation."""

import numpy as np
import pytest

import deltacamera


def rotate_point_deltacamera(x, y, imshape, k):
    """Rotate point according to deltacamera rotate_image90 convention.

    deltacamera uses specific anchor points:
    - k=1: 90 CCW around (a, a) where a = (h-1)/2
    - k=2: 180 around image center
    - k=3: 90 CW around (a, a) where a = (w-1)/2
    """
    h, w = imshape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2

    k = k % 4
    if k == 0:
        return x, y
    elif k == 1:
        a = (h - 1) / 2
        return 2 * a - y, x
    elif k == 2:
        return 2 * cx - x, 2 * cy - y
    else:  # k == 3
        a = (w - 1) / 2
        return y, 2 * a - x


def rotate_points_deltacamera(points, imshape, k):
    """Rotate multiple points according to deltacamera rotate_image90 convention."""
    x, y = points[:, 0], points[:, 1]
    new_x, new_y = rotate_point_deltacamera(x, y, imshape, k)
    return np.stack([new_x, new_y], axis=1).astype(np.float32)


class TestRotateImage90Distortion:
    """Test that rotate_image90 correctly transforms distortion."""

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_rotate90_matches_manual_rotation(self, k):
        """Project then rotate should match rotate_camera then project."""
        # Awkward distortion coefficients
        d = np.array([
            -0.10457,   # k1
            0.03821,    # k2
            0.00173,    # p1
            -0.00289,   # p2
            0.01534,    # k3
        ], dtype=np.float32)

        # Camera at origin with identity rotation (world points = camera points)
        camera = deltacamera.Camera(
            optical_center=[0, 0, 0],
            rot_world_to_cam=np.eye(3, dtype=np.float32),
            intrinsic_matrix=[[517.3, 0, 327.8], [0, 518.9, 241.2], [0, 0, 1]],
            distortion_coeffs=d,
        )
        imshape = (480, 640)

        # Points in front of camera (small angles for valid distortion)
        np.random.seed(42)
        world_points = np.random.randn(50, 3).astype(np.float32) * 0.1
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        # Method 1: Project with original camera, then rotate pixels
        img_orig = camera.world_to_image(world_points)
        img_rotated_manual = rotate_points_deltacamera(img_orig, imshape, k)

        # Method 2: Rotate camera, then project
        rotated_cam = camera.copy()
        rotated_cam.rotate_image90(imshape, k=k)
        img_rotated_cam = rotated_cam.world_to_image(world_points)

        # Filter valid projections
        valid = ~np.isnan(img_orig[:, 0]) & ~np.isnan(img_rotated_cam[:, 0])
        assert np.sum(valid) > 40, "Most points should be valid"

        np.testing.assert_allclose(
            img_rotated_cam[valid],
            img_rotated_manual[valid],
            atol=0.5  # Allow half-pixel tolerance
        )

    def test_rotate90_with_thin_prism(self):
        """Test rotation with full 12-param distortion including thin prism."""
        d = np.array([
            -0.08123,   # k1
            0.02456,    # k2
            0.00134,    # p1
            -0.00278,   # p2
            0.00891,    # k3
            0.01234,    # k4
            -0.00567,   # k5
            0.00345,    # k6
            0.00089,    # s1
            -0.00156,   # s2
            0.00067,    # s3
            -0.00023,   # s4
        ], dtype=np.float32)

        camera = deltacamera.Camera(
            optical_center=[0, 0, 0],
            rot_world_to_cam=np.eye(3, dtype=np.float32),
            intrinsic_matrix=[[523.7, 0, 319.4], [0, 524.1, 239.8], [0, 0, 1]],
            distortion_coeffs=d,
        )
        imshape = (480, 640)

        np.random.seed(123)
        world_points = np.random.randn(30, 3).astype(np.float32) * 0.08
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2.5

        # Project, rotate pixels
        img_orig = camera.world_to_image(world_points)
        img_rotated_manual = rotate_points_deltacamera(img_orig, imshape, k=1)

        # Rotate camera, project
        rotated_cam = camera.copy()
        rotated_cam.rotate_image90(imshape, k=1)
        img_rotated_cam = rotated_cam.world_to_image(world_points)

        valid = ~np.isnan(img_orig[:, 0]) & ~np.isnan(img_rotated_cam[:, 0])

        np.testing.assert_allclose(
            img_rotated_cam[valid],
            img_rotated_manual[valid],
            atol=0.5
        )

    def test_rotate90_k0_identity(self):
        """k=0 should not change projections."""
        d = np.array([-0.11234, 0.04567, 0.00198, -0.00312, 0.01789], dtype=np.float32)
        camera = deltacamera.Camera(
            optical_center=[0, 0, 0],
            rot_world_to_cam=np.eye(3, dtype=np.float32),
            intrinsic_matrix=[[511.2, 0, 325.6], [0, 512.8, 238.4], [0, 0, 1]],
            distortion_coeffs=d,
        )
        imshape = (480, 640)

        np.random.seed(456)
        world_points = np.random.randn(20, 3).astype(np.float32) * 0.1
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        img_orig = camera.world_to_image(world_points)

        rotated_cam = camera.copy()
        rotated_cam.rotate_image90(imshape, k=0)
        img_after = rotated_cam.world_to_image(world_points)

        np.testing.assert_allclose(img_after, img_orig, atol=1e-5)

    def test_rotate90_four_times_returns_original(self):
        """Rotating 4 times should give same projection as original."""
        d = np.array([-0.09876, 0.03214, 0.00145, -0.00267, 0.01432], dtype=np.float32)
        camera = deltacamera.Camera(
            optical_center=[0, 0, 0],
            rot_world_to_cam=np.eye(3, dtype=np.float32),
            intrinsic_matrix=[[505.5, 0, 320.0], [0, 505.5, 240.0], [0, 0, 1]],
            distortion_coeffs=d,
        )
        # Use square image for clean 4x rotation
        imshape = (480, 480)

        np.random.seed(789)
        world_points = np.random.randn(25, 3).astype(np.float32) * 0.1
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        img_orig = camera.world_to_image(world_points)

        rotated_cam = camera.copy()
        for _ in range(4):
            rotated_cam.rotate_image90(imshape, k=1)
        img_after = rotated_cam.world_to_image(world_points)

        valid = ~np.isnan(img_orig[:, 0]) & ~np.isnan(img_after[:, 0])
        np.testing.assert_allclose(img_after[valid], img_orig[valid], atol=1e-3)


class TestRotateImageArbitrary:
    """Test rotate_image with arbitrary angles."""

    def test_rotate_180_matches_k2(self):
        """180 degree rotation should match rotate_image90 with k=2."""
        d = np.array([-0.12345, 0.05678, 0.00234, -0.00345, 0.01567], dtype=np.float32)
        camera = deltacamera.Camera(
            optical_center=[0, 0, 0],
            rot_world_to_cam=np.eye(3, dtype=np.float32),
            intrinsic_matrix=[[509.3, 0, 321.7], [0, 510.1, 239.3], [0, 0, 1]],
            distortion_coeffs=d,
        )
        imshape = (480, 640)

        np.random.seed(111)
        world_points = np.random.randn(30, 3).astype(np.float32) * 0.1
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        # rotate_image with pi
        cam_pi = camera.copy()
        cam_pi.rotate_image(np.pi, imshape)
        img_pi = cam_pi.world_to_image(world_points)

        # rotate_image90 with k=2
        cam_k2 = camera.copy()
        cam_k2.rotate_image90(imshape, k=2)
        img_k2 = cam_k2.world_to_image(world_points)

        valid = ~np.isnan(img_pi[:, 0]) & ~np.isnan(img_k2[:, 0])
        np.testing.assert_allclose(img_k2[valid], img_pi[valid], atol=1e-3)

    def test_rotate_roundtrip(self):
        """Rotate by angle then by -angle should restore original projection."""
        d = np.array([-0.07654, 0.02987, 0.00187, -0.00234, 0.00876], dtype=np.float32)
        camera = deltacamera.Camera(
            optical_center=[0, 0, 0],
            rot_world_to_cam=np.eye(3, dtype=np.float32),
            intrinsic_matrix=[[515.8, 0, 318.9], [0, 516.2, 241.6], [0, 0, 1]],
            distortion_coeffs=d,
        )
        imshape = (480, 640)
        angle = 0.73  # arbitrary angle

        np.random.seed(222)
        world_points = np.random.randn(30, 3).astype(np.float32) * 0.1
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        img_orig = camera.world_to_image(world_points)

        rotated_cam = camera.copy()
        rotated_cam.rotate_image(angle, imshape)
        rotated_cam.rotate_image(-angle, imshape)
        img_after = rotated_cam.world_to_image(world_points)

        valid = ~np.isnan(img_orig[:, 0]) & ~np.isnan(img_after[:, 0])
        np.testing.assert_allclose(img_after[valid], img_orig[valid], atol=1e-3)


class TestRotateImageFisheye:
    """Test that fisheye distortion is handled correctly during rotation."""

    def test_fisheye_rotate90(self):
        """Fisheye cameras should also work with rotate_image90."""
        # Fisheye (4 params = Kannala-Brandt)
        d = np.array([0.08234, -0.01567, 0.00345, -0.00089], dtype=np.float32)
        camera = deltacamera.Camera(
            optical_center=[0, 0, 0],
            rot_world_to_cam=np.eye(3, dtype=np.float32),
            intrinsic_matrix=[[285.4, 0, 319.2], [0, 286.1, 240.8], [0, 0, 1]],
            distortion_coeffs=d,
        )
        imshape = (480, 640)

        np.random.seed(333)
        world_points = np.random.randn(30, 3).astype(np.float32) * 0.15
        world_points[:, 2] = np.abs(world_points[:, 2]) + 1.5

        img_orig = camera.world_to_image(world_points)
        img_rotated_manual = rotate_points_deltacamera(img_orig, imshape, k=1)

        rotated_cam = camera.copy()
        rotated_cam.rotate_image90(imshape, k=1)
        img_rotated_cam = rotated_cam.world_to_image(world_points)

        valid = ~np.isnan(img_orig[:, 0]) & ~np.isnan(img_rotated_cam[:, 0])

        np.testing.assert_allclose(
            img_rotated_cam[valid],
            img_rotated_manual[valid],
            atol=0.5
        )