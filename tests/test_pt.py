"""Tests for deltacamera.pt (PyTorch camera operations)."""

import numpy as np
import pytest
import torch

import deltacamera
import deltacamera.pt as pt
from conftest import random_rotation_matrix, small_rotation_matrix


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _jrdb_camera():
    """Load a real JRDB camera with Brown-Conrady distortion."""
    data = np.load('tests/data/jrdb_calibrations.npz', allow_pickle=True)
    return deltacamera.Camera(
        intrinsic_matrix=data['intrinsic_matrices'][0],
        distortion_coeffs=data['distortion_coeffs'][0],
        image_shape=(480, 752))


def _egohumans_camera():
    """Load a real EgoHumans camera with fisheye distortion."""
    data = np.load('tests/data/egohumans_fisheye_calibrations.npz', allow_pickle=True)
    return deltacamera.Camera(
        intrinsic_matrix=data['intrinsic_matrices'][0],
        distortion_coeffs=data['distortion_coeffs'][0],
        image_shape=(480, 640))


def _to_pt(cam_np):
    return pt.Camera.from_numpy(cam_np).to(_device())


# =========================================================================
# Numpy vs torch consistency: coordinate transforms
# =========================================================================

class TestCoordinateTransformConsistency:
    """pt.Camera coordinate transforms should match numpy deltacamera.Camera."""

    def test_world_to_image_brown_conrady(self):
        cam_cpu = _jrdb_camera()
        cam_t = _to_pt(cam_cpu)

        # Points close to optical axis (small angles, within valid distortion region)
        cam_pts = np.random.randn(200, 3).astype(np.float32) * 0.3
        cam_pts[:, 2] = np.abs(cam_pts[:, 2]) + 3  # well in front
        # These are camera-space points, convert to world (identity R, zero t)
        world_pts = cam_pts
        world_pts_t = torch.from_numpy(world_pts).to(_device())

        cpu_result = cam_cpu.world_to_image(world_pts)
        pt_result = cam_t.world_to_image(world_pts_t).cpu().numpy()

        valid = ~np.isnan(cpu_result[:, 0]) & ~np.isnan(pt_result[:, 0])
        assert valid.sum() > 150
        np.testing.assert_allclose(pt_result[valid], cpu_result[valid], atol=0.05)

    def test_world_to_image_fisheye(self):
        cam_cpu = _egohumans_camera()
        cam_t = _to_pt(cam_cpu)

        cam_pts = np.random.randn(200, 3).astype(np.float32) * 0.5
        cam_pts[:, 2] = np.abs(cam_pts[:, 2]) + 2
        world_pts = cam_pts
        world_pts_t = torch.from_numpy(world_pts).to(_device())

        cpu_result = cam_cpu.world_to_image(world_pts)
        pt_result = cam_t.world_to_image(world_pts_t).cpu().numpy()

        valid = ~np.isnan(cpu_result[:, 0]) & ~np.isnan(pt_result[:, 0])
        assert valid.sum() > 150
        np.testing.assert_allclose(pt_result[valid], cpu_result[valid], atol=0.05)

    def test_image_to_world_roundtrip(self):
        """image_to_world(world_to_image(p)) should recover the ray direction."""
        cam_cpu = _jrdb_camera()
        cam_cpu.R = random_rotation_matrix()
        cam_cpu.t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cam_t = _to_pt(cam_cpu)

        # Start from camera-space points at known depth
        cam_pts = np.random.randn(50, 3).astype(np.float32) * 0.2
        cam_pts[:, 2] = 5.0
        cam_pts_t = torch.from_numpy(cam_pts).to(_device())

        pixels = cam_t.camera_to_image(cam_pts_t)
        recovered = cam_t.image_to_camera(pixels, depth=5.0)

        valid = ~torch.isnan(pixels[:, 0])
        np.testing.assert_allclose(
            recovered[valid].cpu().numpy(), cam_pts[valid.cpu().numpy()],
            atol=0.01)


# =========================================================================
# Image reprojection: numpy vs torch consistency
# =========================================================================

class TestReprojectionConsistency:
    """pt.reproject_image should produce the same result as numpy reprojection."""

    def _compare_reprojection(self, cam_cpu, new_cam_cpu, imshape):
        h, w = imshape
        cam_t = _to_pt(cam_cpu)
        new_cam_t = _to_pt(new_cam_cpu)

        image_np = np.random.rand(h, w, 3).astype(np.float32)
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(_device())

        cpu_result = deltacamera.reproject_image(
            image_np, cam_cpu, new_cam_cpu, output_imshape=(h, w))
        pt_result = pt.reproject_image(
            image_t, cam_t, new_cam_t, (h, w)
        ).squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Mask to pixels where both produce valid output
        valid = (cpu_result.sum(axis=2) > 0) & (pt_result.sum(axis=2) > 0)
        assert valid.sum() > h * w * 0.3, 'Too few valid pixels'
        np.testing.assert_allclose(
            pt_result[valid], cpu_result[valid], atol=2.0,
            err_msg='torch and numpy reprojection differ significantly')

    def test_undistort_brown_conrady(self):
        cam = _jrdb_camera()
        self._compare_reprojection(cam, cam.undistort(), (480, 752))

    def test_undistort_fisheye(self):
        cam = _egohumans_camera()
        self._compare_reprojection(cam, cam.undistort(), (480, 640))

    def test_rotation_only(self):
        cam = _jrdb_camera()
        new_cam = cam.copy()
        new_cam.R = small_rotation_matrix(10) @ cam.R
        self._compare_reprojection(cam, new_cam, (480, 752))

    def test_zoom(self):
        cam = _jrdb_camera()
        new_cam = cam.zoom(1.5)
        self._compare_reprojection(cam, new_cam, (480, 752))


# =========================================================================
# Undistorting video: batch of identical frames
# =========================================================================

class TestBatchUndistortion:
    """Batched reproject_image (same cameras, multiple frames)."""

    def test_batch_matches_single(self):
        """Batched output should be identical to processing frames individually."""
        cam = _to_pt(_jrdb_camera())
        cam_undist = cam.undistorted()
        B = 4
        images = torch.randn(B, 3, 480, 752, device=_device())

        batched = pt.reproject_image(images, cam, cam_undist, (480, 752))

        for i in range(B):
            single = pt.reproject_image(
                images[i:i + 1], cam, cam_undist, (480, 752))
            torch.testing.assert_close(batched[i:i + 1], single)

    def test_output_shape(self):
        cam = _to_pt(_jrdb_camera())
        cam_undist = cam.undistorted()
        images = torch.randn(8, 3, 480, 752, device=_device())

        result = pt.reproject_image(images, cam, cam_undist, (400, 400))
        assert result.shape == (8, 3, 400, 400)

    def test_undistorted_preserves_center(self):
        """Center pixel should not move much when undistorting."""
        cam = _to_pt(_jrdb_camera())
        cam_undist = cam.undistorted()

        # Image with a bright dot at center
        h, w = 480, 752
        image = torch.zeros(1, 1, h, w, device=_device())
        cy, cx = h // 2, w // 2
        image[0, 0, cy - 2:cy + 3, cx - 2:cx + 3] = 1.0

        result = pt.reproject_image(image, cam, cam_undist, (h, w))

        # Find peak in result
        flat = result[0, 0]
        peak_y, peak_x = divmod(flat.argmax().item(), w)
        assert abs(peak_y - cy) < 5 and abs(peak_x - cx) < 5


# =========================================================================
# reproject_image_multi: per-frame camera variation
# =========================================================================

class TestReprojectMulti:
    """reproject_image_multi with different cameras per frame."""

    def test_multi_matches_loop(self):
        """Multi should give same results as looping reproject_image."""
        cam_cpu = _jrdb_camera()
        cam_base = _to_pt(cam_cpu)
        B = 4
        images = torch.randn(B, 3, 480, 752, device=_device())

        # Different rotation per frame
        old_cams = []
        new_cams = []
        for i in range(B):
            R = torch.from_numpy(
                small_rotation_matrix(3 * (i + 1))).to(_device())
            old_i = cam_base.copy(R=R @ cam_base.R)
            old_cams.append(old_i)
            new_cams.append(cam_base.undistorted())

        multi_result = pt.reproject_image_multi(
            images, old_cams, new_cams, (480, 752))

        for i in range(B):
            single = pt.reproject_image(
                images[i:i + 1], old_cams[i], new_cams[i], (480, 752))
            torch.testing.assert_close(multi_result[i:i + 1], single, atol=0.01, rtol=1e-4)

    def test_broadcast_single_camera(self):
        """Passing a single camera should broadcast to all batch elements."""
        cam = _to_pt(_jrdb_camera())
        cam_undist = cam.undistorted()
        B = 3
        images = torch.randn(B, 3, 480, 752, device=_device())

        multi_result = pt.reproject_image_multi(images, cam, cam_undist, (480, 752))
        single_result = pt.reproject_image(images, cam, cam_undist, (480, 752))

        torch.testing.assert_close(multi_result, single_result)

    def test_mixed_distortion_old_varies(self):
        """Batched path when old cameras have different distortion (new shared)."""
        cam_bc = _to_pt(_jrdb_camera())
        cam_fe = _to_pt(_egohumans_camera())
        # Both undistort to pinhole → new_d is shared (None)
        new_cam = cam_bc.undistorted().copy(image_shape=(240, 320))
        B = 2
        images = torch.randn(B, 3, 240, 320, device=_device())

        old_cams = [cam_bc.copy(image_shape=(240, 320)),
                    cam_fe.copy(image_shape=(240, 320))]
        new_cams = [new_cam, new_cam]

        multi_result = pt.reproject_image_multi(images, old_cams, new_cams, (240, 320))
        for i in range(B):
            single = pt.reproject_image(
                images[i:i + 1], old_cams[i], new_cams[i], (240, 320))
            torch.testing.assert_close(multi_result[i:i + 1], single, atol=0.01, rtol=1e-4)

    def test_mixed_distortion_new_varies(self):
        """Batched path when new cameras have different distortion (old shared)."""
        cam_bc = _to_pt(_jrdb_camera())
        cam_fe = _to_pt(_egohumans_camera())
        # Same old camera, different new cameras
        old_cam = cam_bc.undistorted().copy(image_shape=(240, 320))
        B = 2
        images = torch.randn(B, 3, 240, 320, device=_device())

        new_cams = [cam_bc.copy(image_shape=(240, 320)),
                    cam_fe.copy(image_shape=(240, 320))]

        multi_result = pt.reproject_image_multi(images, old_cam, new_cams, (240, 320))
        for i in range(B):
            single = pt.reproject_image(
                images[i:i + 1], old_cam, new_cams[i], (240, 320))
            torch.testing.assert_close(multi_result[i:i + 1], single, atol=0.01, rtol=1e-4)

    def test_single_image_auto_expand(self):
        """Single image (B=1) should auto-expand to match multiple cameras."""
        cam = _to_pt(_jrdb_camera())
        N = 3
        image = torch.randn(1, 3, 480, 752, device=_device())

        new_cams = [cam.rotated(yaw=0.02 * i).undistorted() for i in range(N)]
        result = pt.reproject_image_multi(image, cam, new_cams, (480, 752))
        assert result.shape == (N, 3, 480, 752)

        for i in range(N):
            single = pt.reproject_image(image, cam, new_cams[i], (480, 752))
            torch.testing.assert_close(result[i:i + 1], single, atol=0.01, rtol=1e-4)


# =========================================================================
# Perspective-aware cropping
# =========================================================================

class TestPerspectiveCrop:
    """Cropping via turned_towards + zoom."""

    def test_crop_preserves_target_point(self):
        """After turning towards a world point, it should appear at image center."""
        cam = _to_pt(_jrdb_camera())
        cam_undist = cam.undistorted()

        target_world = torch.tensor([0.5, -0.3, 5.0], device=_device())
        crop_cam = cam_undist.turned_towards(target_world_point=target_world)
        crop_cam = crop_cam.zoomed(2.0)
        crop_cam = crop_cam.copy(image_shape=(256, 256))
        crop_cam = crop_cam.principal_point_centered()

        pixel = crop_cam.world_to_image(target_world.unsqueeze(0))
        px, py = pixel[0, 0].item(), pixel[0, 1].item()

        # Should be near center (127.5, 127.5)
        assert abs(px - 127.5) < 2 and abs(py - 127.5) < 2

    def test_crop_warp_runs(self):
        """Full crop pipeline should produce valid output."""
        cam = _to_pt(_jrdb_camera())
        target = torch.tensor([0.0, 0.0, 5.0], device=_device())
        crop_cam = cam.turned_towards(target_world_point=target)
        crop_cam = crop_cam.zoomed(1.5)
        crop_cam = crop_cam.copy(image_shape=(256, 256))

        image = torch.randn(1, 3, 480, 752, device=_device())
        result = pt.reproject_image(image, cam, crop_cam, (256, 256))
        assert result.shape == (1, 3, 256, 256)
        assert not torch.isnan(result).all()


# =========================================================================
# Camera manipulation
# =========================================================================

class TestCameraManipulation:
    """Camera manipulation methods preserve expected properties."""

    def test_undistorted_has_no_distortion(self):
        cam = _to_pt(_jrdb_camera())
        assert cam.has_distortion()
        cam_undist = cam.undistorted()
        assert not cam_undist.has_distortion()
        assert cam_undist.d is None

    def test_image_scaled_preserves_fov(self):
        """Scaling the image should not change what is visible, just resolution."""
        cam = _to_pt(_jrdb_camera())
        cam2 = cam.image_scaled(2.0)

        # A world point should project to 2x the pixel coordinates
        p = torch.tensor([[0.3, -0.2, 5.0]], device=_device())
        px1 = cam.world_to_image(p)
        px2 = cam2.world_to_image(p)
        np.testing.assert_allclose(
            px2.cpu().numpy(), px1.cpu().numpy() * 2, atol=0.1)

    def test_zoomed_changes_fov(self):
        """Zooming should scale projected coordinates."""
        cam = _to_pt(_jrdb_camera())
        cam2 = cam.zoomed(2.0)

        p = torch.tensor([[0.1, -0.05, 5.0]], device=_device())
        px1 = cam.camera_to_image(p)
        px2 = cam2.camera_to_image(p)

        # Offset from principal point should be 2x
        pp = cam.K[:2, 2]
        offset1 = px1[0] - pp
        offset2 = px2[0] - pp
        np.testing.assert_allclose(
            offset2.cpu().numpy(), offset1.cpu().numpy() * 2, atol=0.1)

    def test_rotated_yaw_shifts_image(self):
        """Small yaw rotation should shift world points in the image."""
        cam = _to_pt(_jrdb_camera())
        cam2 = cam.rotated(yaw=0.05)

        # A world point slightly off-center
        p = torch.tensor([[0.2, 0.0, 5.0]], device=_device())
        px1 = cam.world_to_image(p)
        px2 = cam2.world_to_image(p)

        # Yaw should shift mostly in x
        dx = abs(px2[0, 0].item() - px1[0, 0].item())
        dy = abs(px2[0, 1].item() - px1[0, 1].item())
        assert dx > 5, 'Yaw rotation should shift x'
        assert dy < dx, 'Yaw should mostly affect x, not y'

    def test_image_hflipped_mirrors_projection(self):
        """image_hflipped should mirror pixel coordinates."""
        cam = _to_pt(_jrdb_camera()).undistorted()
        cam_flip = cam.image_hflipped()

        p = torch.tensor([[0.1, 0.0, 5.0]], device=_device())
        px = cam.world_to_image(p)
        px_flip = cam_flip.world_to_image(p)

        # x should be mirrored: new_x ≈ (W-1) - old_x
        w = cam.image_shape[1]
        np.testing.assert_allclose(
            px_flip[0, 0].cpu().numpy(),
            (w - 1) - px[0, 0].cpu().numpy(), atol=1.0)

    def test_copy_independence(self):
        """Modifying a copy should not affect the original."""
        cam = _to_pt(_jrdb_camera())
        R_orig = cam.R.clone()
        R_new = torch.tensor(
            small_rotation_matrix(30), device=_device(), dtype=torch.float32)
        cam2 = cam.copy(R=R_new)
        # Original should be unchanged
        torch.testing.assert_close(cam.R, R_orig)
        # Copy should have the new R
        torch.testing.assert_close(cam2.R, R_new)

    def test_to_device_roundtrip(self):
        cam = _to_pt(_jrdb_camera())
        cam_cpu = cam.to('cpu')
        cam_t2 = cam_cpu.to(_device())
        torch.testing.assert_close(cam_t2.K, cam.K)
        torch.testing.assert_close(cam_t2.d, cam.d)


# =========================================================================
# Point reprojection
# =========================================================================

class TestPointReprojection:
    """Point reprojection between cameras."""

    def test_identity(self):
        cam = _to_pt(_jrdb_camera())
        points = torch.tensor(
            [[376, 240], [200, 100], [500, 400]], device=_device(), dtype=torch.float32)
        result = pt.reproject_image_points(points, cam, cam)
        torch.testing.assert_close(result, points, atol=0.1, rtol=1e-4)

    def test_roundtrip(self):
        cam1 = _to_pt(_jrdb_camera())
        cam2 = cam1.rotated(yaw=0.05, pitch=0.02)

        points = torch.tensor(
            [[376, 240], [300, 200], [450, 350]], device=_device(), dtype=torch.float32)
        fwd = pt.reproject_image_points(points, cam1, cam2)
        back = pt.reproject_image_points(fwd, cam2, cam1)

        valid = ~torch.isnan(back[:, 0])
        torch.testing.assert_close(back[valid], points[valid], atol=0.5, rtol=1e-3)

    def test_consistency_with_cpu(self):
        cam1_cpu = _jrdb_camera()
        cam2_cpu = cam1_cpu.copy()
        cam2_cpu.R = small_rotation_matrix(5) @ cam1_cpu.R

        cam1_gpu = _to_pt(cam1_cpu)
        cam2_gpu = _to_pt(cam2_cpu)

        points_np = np.array([[376, 240], [300, 200], [450, 300]], dtype=np.float32)
        points_t = torch.from_numpy(points_np).to(_device())

        cpu_result = deltacamera.reprojection.reproject_image_points(
            points_np, cam1_cpu, cam2_cpu)
        pt_result = pt.reproject_image_points(points_t, cam1_gpu, cam2_gpu).cpu().numpy()

        valid = ~np.isnan(cpu_result[:, 0]) & ~np.isnan(pt_result[:, 0])
        np.testing.assert_allclose(pt_result[valid], cpu_result[valid], atol=0.5)


# =========================================================================
# Depth map reprojection
# =========================================================================

class TestDepthReprojection:
    """Depth map reprojection with z-correction."""

    def test_identity(self):
        cam = _to_pt(_jrdb_camera())
        depth = torch.ones(1, 1, 480, 752, device=_device()) * 5.0
        result = pt.reproject_depth_map(depth, cam, cam, (480, 752))
        # Center region should be preserved (borders may have NaN/0)
        center = result[0, 0, 100:380, 100:650]
        valid = center[center > 0]
        assert valid.numel() > 0
        np.testing.assert_allclose(valid.cpu().numpy(), 5.0, atol=0.1)

    def test_rotation_changes_depth(self):
        """Rotating should change depth values via z-correction."""
        cam = _to_pt(_jrdb_camera())
        cam2 = cam.rotated(yaw=0.1)

        depth = torch.ones(1, 1, 480, 752, device=_device()) * 5.0
        result = pt.reproject_depth_map(depth, cam, cam2, (480, 752))

        # The reprojected depth should vary (not all 5.0)
        valid = result[result > 0]
        assert valid.std().item() > 0.01


# =========================================================================
# precomp_undist_maps flag
# =========================================================================

class TestPrecompUndistMaps:
    """precomp_undist_maps should give results close to the exact solver."""

    def test_precomp_close_to_exact_bc(self):
        cam = _to_pt(_jrdb_camera())
        cam2 = cam.rotated(yaw=0.05)
        image = torch.randn(1, 3, 480, 752, device=_device())

        exact = pt.reproject_image(
            image, cam, cam2, (480, 752), precomp_undist_maps=False)
        precomp = pt.reproject_image(
            image, cam, cam2, (480, 752), precomp_undist_maps=True)

        # Should be very close (LUT introduces tiny interpolation error)
        valid = (exact.abs().sum(dim=1, keepdim=True) > 0) & \
                (precomp.abs().sum(dim=1, keepdim=True) > 0)
        diff = (exact - precomp).abs()
        diff_valid = diff[valid.expand_as(diff)]
        assert diff_valid.max().item() < 5.0, 'Precomp and exact differ too much'
        assert diff_valid.mean().item() < 0.5, 'Mean difference too high'

    def test_precomp_close_to_exact_fisheye(self):
        cam = _to_pt(_egohumans_camera())
        cam2 = cam.rotated(yaw=0.05)
        image = torch.randn(1, 3, 480, 640, device=_device())

        exact = pt.reproject_image(
            image, cam, cam2, (480, 640), precomp_undist_maps=False)
        precomp = pt.reproject_image(
            image, cam, cam2, (480, 640), precomp_undist_maps=True)

        valid = (exact.abs().sum(dim=1, keepdim=True) > 0) & \
                (precomp.abs().sum(dim=1, keepdim=True) > 0)
        diff = (exact - precomp).abs()
        diff_valid = diff[valid.expand_as(diff)]
        assert diff_valid.max().item() < 5.0
        assert diff_valid.mean().item() < 0.5


# =========================================================================
# Antialiasing
# =========================================================================

class TestAntialiasing:
    """Antialiasing via supersampling should smooth the result."""

    def test_antialias_smoother_than_plain(self):
        """Antialiased result should have less high-frequency content."""
        cam = _to_pt(_jrdb_camera())
        cam_undist = cam.undistorted()

        # Checkerboard pattern (worst case for aliasing)
        h, w = 480, 752
        checker = torch.zeros(1, 1, h, w, device=_device())
        checker[0, 0, ::2, ::2] = 1.0
        checker[0, 0, 1::2, 1::2] = 1.0

        plain = pt.reproject_image(checker, cam, cam_undist, (h, w), antialias_factor=1)
        smooth = pt.reproject_image(checker, cam, cam_undist, (h, w), antialias_factor=2)

        # Compute high-frequency energy (Laplacian variance)
        def hf_energy(img):
            k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                             device=_device(), dtype=torch.float32).reshape(1, 1, 3, 3)
            lap = torch.nn.functional.conv2d(img, k, padding=1)
            return lap.var().item()

        assert hf_energy(smooth) < hf_energy(plain)


# =========================================================================
# from_numpy conversion
# =========================================================================

class TestFromNumpy:
    """Camera.from_numpy should faithfully transfer all parameters."""

    def test_parameters_match(self):
        cam_cpu = _jrdb_camera()
        cam_cpu.R = random_rotation_matrix()
        cam_cpu.t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cam_t = pt.Camera.from_numpy(cam_cpu)

        np.testing.assert_allclose(cam_t.K.numpy(), cam_cpu.intrinsic_matrix, atol=1e-6)
        np.testing.assert_allclose(cam_t.R.numpy(), cam_cpu.R, atol=1e-6)
        np.testing.assert_allclose(cam_t.t.numpy(), cam_cpu.t, atol=1e-6)
        assert cam_t.image_shape == cam_cpu.image_shape

    def test_distortion_preserved(self):
        cam_cpu = _jrdb_camera()
        cam_t = pt.Camera.from_numpy(cam_cpu)
        assert cam_t.has_distortion()
        assert not cam_t.has_fisheye_distortion()
        assert cam_t.has_nonfisheye_distortion()

    def test_fisheye_detected(self):
        cam_cpu = _egohumans_camera()
        cam_t = pt.Camera.from_numpy(cam_cpu)
        assert cam_t.has_fisheye_distortion()
        assert not cam_t.has_nonfisheye_distortion()

    def test_no_distortion(self):
        cam_cpu = deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]])
        cam_t = pt.Camera.from_numpy(cam_cpu)
        assert not cam_t.has_distortion()
        assert cam_t.d is None


# =========================================================================
# Image geometry: hflip, rotation, scale consistency
# =========================================================================

class TestImageGeometry:
    """Image geometry adjustments should be consistent with reprojection."""

    def test_hflip_reprojects_correctly(self):
        """Flipping image + adjusting camera should match reprojection of original."""
        cam = _to_pt(_jrdb_camera())
        cam_flip = cam.image_hflipped()

        p = torch.tensor([[200.0, 300.0]], device=_device())
        p_flip = pt.reproject_image_points(p, cam, cam_flip)

        # x should be mirrored: new_x = (W-1) - old_x (approximately)
        w = cam.image_shape[1]
        expected_x = w - 1 - p[0, 0].item()
        assert abs(p_flip[0, 0].item() - expected_x) < 2

    def test_image_resized_consistency(self):
        cam = _to_pt(_jrdb_camera())
        cam2 = cam.image_resized((240, 376))

        p_world = torch.tensor([[0.1, -0.05, 5.0]], device=_device())
        px1 = cam.world_to_image(p_world)
        px2 = cam2.world_to_image(p_world)

        # Pixels should scale by the resize ratio
        np.testing.assert_allclose(
            px2[0, 0].cpu().item(), px1[0, 0].cpu().item() * 376 / 752, atol=0.5)
        np.testing.assert_allclose(
            px2[0, 1].cpu().item(), px1[0, 1].cpu().item() * 240 / 480, atol=0.5)


# =========================================================================
# Gradient flow
# =========================================================================

class TestGradientFlow:
    """Verify gradients flow through the reprojection pipeline."""

    def test_gradient_through_rotation(self):
        """Gradients should flow from reprojected image to rotation parameters."""
        cam = _to_pt(_jrdb_camera())
        cam_undist = cam.undistorted()

        # Make rotation a leaf that requires grad
        R = cam.R.clone().detach().requires_grad_(True)
        cam_r = cam.copy(R=R)

        image = torch.randn(1, 3, 480, 752, device=_device())
        result = pt.reproject_image(
            image, cam_r, cam_undist, (480, 752), precomp_undist_maps=False)
        loss = result.sum()
        loss.backward()

        assert R.grad is not None
        assert R.grad.abs().sum().item() > 0

    def test_gradient_through_intrinsics(self):
        """Gradients should flow from reprojected image to K."""
        cam = _to_pt(_jrdb_camera())

        K = cam.K.clone().detach().requires_grad_(True)
        cam_k = cam.copy(K=K, d=None)  # no distortion for simplicity
        cam_target = cam.copy(d=None)

        image = torch.randn(1, 3, 480, 752, device=_device())
        result = pt.reproject_image(
            image, cam_k, cam_target, (480, 752), precomp_undist_maps=False)
        loss = result.sum()
        loss.backward()

        assert K.grad is not None
        assert K.grad.abs().sum().item() > 0
