
import math

import numpy as np
import torch

from . import coordframes
from . import distortion as dist_module
from . import reprojection as reproj_module
from . import validity as validity_module

_UNSET = object()


class Camera:
    """Camera with parameters stored as torch tensors for GPU-accelerated operations.

    Attributes:
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix (world-to-camera)
        t: (3,) camera position in world coordinates
        d: (C,) distortion coefficients, or None
        image_shape: (h, w) or None
        world_up: (3,) world up direction vector (normalized)
    """

    def __init__(self, K, R=None, t=None, d=None, image_shape=None, world_up=None):
        """Create a GPU Camera.

        Args:
            K: (3, 3) intrinsic matrix (tensor or array)
            R: (3, 3) rotation matrix, defaults to identity
            t: (3,) camera position, defaults to origin
            d: distortion coefficients, or None
            image_shape: (h, w) or None
            world_up: (3,) world up direction, defaults to (0, 0, 1)
        """
        self.K = _to_tensor(K)
        device = self.K.device
        dtype = self.K.dtype
        self.R = _to_tensor(R, device, dtype) if R is not None else torch.eye(
            3, device=device, dtype=dtype)
        self.t = _to_tensor(t, device, dtype) if t is not None else torch.zeros(
            3, device=device, dtype=dtype)
        self.d = _to_tensor(d, device, dtype) if d is not None else None
        self.image_shape = tuple(image_shape) if image_shape is not None else None
        if world_up is not None:
            wu = _to_tensor(world_up, device, dtype)
            self.world_up = wu / torch.linalg.norm(wu)
        else:
            self.world_up = torch.tensor([0, 0, 1], device=device, dtype=dtype)

    @staticmethod
    def from_numpy(camera, device=None):
        """Create a torch Camera from a numpy-based deltacamera.Camera object.

        Args:
            camera: deltacamera.Camera
            device: torch device (default: cpu)

        Returns:
            pt.Camera
        """
        K = torch.from_numpy(np.asarray(camera.intrinsic_matrix, dtype=np.float32))
        R = torch.from_numpy(np.asarray(camera.R, dtype=np.float32))
        t = torch.from_numpy(np.asarray(camera.t, dtype=np.float32))
        world_up = torch.from_numpy(np.asarray(camera.world_up, dtype=np.float32))
        d = None
        if camera.has_distortion():
            d = torch.from_numpy(np.array(camera._distortion_model.coeffs, dtype=np.float32))

        if device is not None:
            K = K.to(device)
            R = R.to(device)
            t = t.to(device)
            world_up = world_up.to(device)
            if d is not None:
                d = d.to(device)

        return Camera(K, R, t, d, camera.image_shape, world_up)

    def to(self, device):
        """Move all tensors to a device. Returns a new Camera."""
        return Camera(
            self.K.to(device), self.R.to(device), self.t.to(device),
            self.d.to(device) if self.d is not None else None,
            self.image_shape, self.world_up.to(device))

    @property
    def device(self):
        return self.K.device

    def has_distortion(self):
        return self.d is not None

    def has_fisheye_distortion(self):
        return self.d is not None and self.d.shape[-1] == 4

    def has_nonfisheye_distortion(self):
        return self.d is not None and self.d.shape[-1] != 4

    # =================================================================
    # Coordinate transforms
    # =================================================================

    def world_to_camera(self, points):
        """Transform world coordinates to camera coordinates.

        Args:
            points: (..., 3) world coordinates

        Returns:
            (..., 3) camera coordinates
        """
        return coordframes.world_to_camera(points, self.R, self.t)

    def camera_to_world(self, points):
        """Transform camera coordinates to world coordinates.

        Args:
            points: (..., 3) camera coordinates

        Returns:
            (..., 3) world coordinates
        """
        return coordframes.camera_to_world(points, self.R, self.t)

    def camera_to_image(self, points):
        """Project camera coordinates to pixel coordinates (with distortion).

        Args:
            points: (..., 3) camera coordinates

        Returns:
            (..., 2) pixel coordinates
        """
        norm_pts = coordframes.project(points)
        norm_pts = self._distort(norm_pts)
        return coordframes.apply_intrinsics(norm_pts, self.K)

    def world_to_image(self, points):
        """Project world coordinates to pixel coordinates (with distortion).

        Args:
            points: (..., 3) world coordinates

        Returns:
            (..., 2) pixel coordinates
        """
        cam_pts = self.world_to_camera(points)
        return self.camera_to_image(cam_pts)

    def image_to_camera(self, points, depth=1.0):
        """Backproject pixel coordinates to camera coordinates (with undistortion).

        Args:
            points: (..., 2) pixel coordinates
            depth: scalar or (...) depth values

        Returns:
            (..., 3) camera coordinates
        """
        norm_pts = coordframes.undo_intrinsics(points, self.K)
        norm_pts = self._undistort(norm_pts)
        return coordframes.backproject(
            norm_pts, torch.eye(3, device=self.device, dtype=self.K.dtype), depth)

    def image_to_world(self, points, depth=1.0):
        """Backproject pixel coordinates to world coordinates (with undistortion).

        Args:
            points: (..., 2) pixel coordinates
            depth: scalar or (...) depth values

        Returns:
            (..., 3) world coordinates
        """
        cam_pts = self.image_to_camera(points, depth)
        return self.camera_to_world(cam_pts)

    # =================================================================
    # Distortion helpers
    # =================================================================

    def _get_valid_region(self):
        if self.d is None:
            return None
        if self.has_fisheye_distortion():
            return validity_module.fisheye_valid_r_max(self.d)
        d_padded = dist_module._pad_distortion_coeffs(self.d, 14)
        return validity_module.brown_conrady_valid_region(d_padded)

    def _distort(self, norm_pts):
        if self.d is None:
            return norm_pts
        if self.has_fisheye_distortion():
            ru, _ = validity_module.fisheye_valid_r_max(self.d)
            return dist_module.distort_fisheye(norm_pts, self.d, ru_valid=ru)
        d_padded = dist_module._pad_distortion_coeffs(self.d, 14)
        vr = validity_module.brown_conrady_valid_region(d_padded)
        return dist_module.distort_brown_conrady(norm_pts, d_padded, valid_region=vr)

    def _undistort(self, norm_pts):
        if self.d is None:
            return norm_pts
        if self.has_fisheye_distortion():
            ru, rd = validity_module.fisheye_valid_r_max(self.d)
            return dist_module.undistort_fisheye(norm_pts, self.d, ru, rd)
        d_padded = dist_module._pad_distortion_coeffs(self.d, 14)
        vr = validity_module.brown_conrady_valid_region(d_padded)
        return dist_module.undistort_brown_conrady(norm_pts, d_padded, vr)

    # =================================================================
    # Camera manipulation (returns new Camera)
    # =================================================================

    def copy(
            self, *, K=_UNSET, R=_UNSET, t=_UNSET, d=_UNSET, image_shape=_UNSET,
            world_up=_UNSET):
        """Create a copy of this camera, optionally with replaced attributes.

        Args:
            K: New intrinsic matrix
            R: New rotation matrix
            t: New camera position
            d: New distortion coefficients (pass None to remove distortion)
            image_shape: New image shape (h, w), or None to unset
            world_up: New world up direction

        Returns:
            New pt.Camera instance
        """
        return Camera(
            K=K if K is not _UNSET else self.K.clone(),
            R=R if R is not _UNSET else self.R.clone(),
            t=t if t is not _UNSET else self.t.clone(),
            d=d if d is not _UNSET else (self.d.clone() if self.d is not None else None),
            image_shape=image_shape if image_shape is not _UNSET else self.image_shape,
            world_up=world_up if world_up is not _UNSET else self.world_up.clone(),
        )

    def undistorted(self, square_pixels=False, zero_skew=False):
        """Return a new camera with distortion removed.

        Args:
            square_pixels: If True, adjust intrinsics for square pixels
            zero_skew: If True, set the skew coefficient to zero

        Returns:
            New Camera without distortion
        """
        new_K = self.K.clone()
        if square_pixels:
            fx, fy = new_K[0, 0], new_K[1, 1]
            fmean = 0.5 * (fx + fy)
            multiplier = torch.eye(3, device=self.device, dtype=self.K.dtype)
            multiplier[0, 0] = fmean / fx
            multiplier[1, 1] = fmean / fy
            new_K = multiplier @ new_K
        if zero_skew:
            new_K[0, 1] = 0
        return self.copy(K=new_K, d=None)

    def image_scaled(self, factor, center_subpixels=False):
        """Return a camera adjusted for a uniformly scaled image.

        Args:
            factor: Scale factor (>1 makes image larger)
            center_subpixels: If True, shift the principal point by (factor-1)/2

        Returns:
            New Camera for the scaled image
        """
        new_K = self.K.clone()
        new_K[:2] *= factor
        if center_subpixels:
            new_K[:2, 2] += (factor - 1) / 2
        new_shape = None
        if self.image_shape is not None:
            new_shape = (
                int(self.image_shape[0] * factor),
                int(self.image_shape[1] * factor))
        return self.copy(K=new_K, image_shape=new_shape)

    def zoomed(self, factor):
        """Return a new camera with focal length scaled by factor."""
        new_K = self.K.clone()
        new_K[:2, :2] *= factor
        return self.copy(K=new_K)

    def rotated(self, yaw=0, pitch=0, roll=0):
        """Return a new camera rotated by yaw, pitch, roll (radians, YXZ intrinsic order)."""
        device, dtype = self.device, self.K.dtype
        yaw = torch.as_tensor(yaw, device=device, dtype=dtype)
        pitch = torch.as_tensor(pitch, device=device, dtype=dtype)
        roll = torch.as_tensor(roll, device=device, dtype=dtype)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cr, sr = torch.cos(roll), torch.sin(roll)
        z = torch.zeros((), device=device, dtype=dtype)
        o = torch.ones((), device=device, dtype=dtype)
        Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy]).reshape(3, 3)
        Rx = torch.stack([o, z, z, z, cp, -sp, z, sp, cp]).reshape(3, 3)
        Rz = torch.stack([cr, -sr, z, sr, cr, z, z, z, o]).reshape(3, 3)
        camera_rotation = Ry @ Rx @ Rz
        new_R = camera_rotation.T @ self.R
        return self.copy(R=new_R)

    def turned_towards(
            self, target_image_point=None, target_world_point=None,
            target_cam_point=None):
        """Return a new camera with optical axis pointing at a target point.

        Exactly one of the three target arguments must be provided.

        Args:
            target_image_point: Image coordinates of the target point
            target_world_point: World coordinates of the target point
            target_cam_point: Camera coordinates of the target point

        Returns:
            New Camera pointing at target
        """
        if target_image_point is not None:
            target_world_point = self.image_to_world(target_image_point)
        elif target_cam_point is not None:
            target_world_point = self.camera_to_world(target_cam_point)

        target_world_point = torch.as_tensor(
            target_world_point, device=self.device, dtype=self.K.dtype)

        new_z = _unit_vec(target_world_point - self.t)
        new_x = _unit_vec(torch.cross(new_z, self.world_up, dim=-1))
        if not torch.all(torch.isfinite(new_x)):
            fallback = torch.tensor([1, 0, 0], device=self.device, dtype=self.K.dtype)
            new_x = _unit_vec(torch.cross(new_z, fallback, dim=-1))
            if not torch.all(torch.isfinite(new_x)):
                fallback = torch.tensor([0, 1, 0], device=self.device, dtype=self.K.dtype)
                new_x = _unit_vec(torch.cross(new_z, fallback, dim=-1))
        new_y = torch.cross(new_z, new_x, dim=-1)

        new_R = torch.stack([new_x, new_y, new_z])
        return self.copy(R=new_R)

    def orbited_around(self, world_point, angle, axis="vertical"):
        """Return a new camera orbited around a world point.

        Args:
            world_point: World coordinates of the pivot point
            angle: Rotation angle in radians
            axis: 'vertical' or 'horizontal'

        Returns:
            New Camera at orbited position
        """
        world_point = torch.as_tensor(
            world_point, device=self.device, dtype=self.K.dtype)

        if axis == "vertical":
            axis_vec = self.world_up
        else:
            lookdir = self.R[2]
            axis_vec = _unit_vec(torch.cross(lookdir, self.world_up, dim=-1))

        rot_matrix = _rodrigues(axis_vec * angle)
        new_t = rot_matrix @ (self.t - world_point) + world_point
        new_R = self.R @ rot_matrix.T

        return self.copy(R=new_R, t=new_t)

    def rolled_upright(self):
        """Return a new camera rolled upright to align with world up vector."""
        new_R = self.R.clone()
        new_x = _unit_vec(torch.cross(new_R[2], self.world_up, dim=-1))
        if not torch.all(torch.isfinite(new_x)):
            return self.copy()
        new_R[0] = new_x
        new_R[1] = -torch.cross(new_R[0], new_R[2], dim=-1)
        return self.copy(R=new_R)

    def hflipped(self):
        """Return a new camera with horizontal flip (negated first row of rotation)."""
        new_R = self.R.clone()
        new_R[0] *= -1
        return self.copy(R=new_R)

    def image_shifted(self, offset):
        """Return a new camera with principal point shifted by offset (x, y)."""
        offset = torch.as_tensor(offset, device=self.device, dtype=self.K.dtype)
        new_K = self.K.clone()
        new_K[:2, 2] += offset
        return self.copy(K=new_K)

    def point_shifted_to(self, current_point, target_point):
        """Return a camera with principal point adjusted to move a point."""
        current_point = torch.as_tensor(
            current_point, device=self.device, dtype=self.K.dtype)
        target_point = torch.as_tensor(
            target_point, device=self.device, dtype=self.K.dtype)
        return self.image_shifted(target_point - current_point)

    def point_shifted_to_center(self, point):
        """Return a camera with principal point adjusted so that point appears at image center."""
        if self.image_shape is None:
            raise ValueError("image_shape must be set for point_shifted_to_center")
        center = torch.tensor(
            [(self.image_shape[1] - 1) / 2, (self.image_shape[0] - 1) / 2],
            device=self.device, dtype=self.K.dtype)
        return self.point_shifted_to(point, center)

    def image_cropped(self, new_shape, anchor=(0, 0)):
        """Return a camera adjusted for a cropped image.

        Args:
            new_shape: New image shape (height, width)
            anchor: Top-left corner of crop in original image (x, y)
        """
        anchor = torch.as_tensor(anchor, device=self.device, dtype=self.K.dtype)
        new_K = self.K.clone()
        new_K[0, 2] -= anchor[0]
        new_K[1, 2] -= anchor[1]
        return self.copy(K=new_K, image_shape=tuple(new_shape))

    def image_padded(self, new_shape, anchor=(0, 0)):
        """Return a camera adjusted for a padded image.

        Args:
            new_shape: New image shape (height, width)
            anchor: Position of original image within padded image (x, y)
        """
        anchor = torch.as_tensor(anchor, device=self.device, dtype=self.K.dtype)
        new_K = self.K.clone()
        new_K[0, 2] += anchor[0]
        new_K[1, 2] += anchor[1]
        return self.copy(K=new_K, image_shape=tuple(new_shape))

    def image_resized(self, new_shape):
        """Return a camera adjusted for a resized image."""
        if self.image_shape is None:
            raise ValueError("image_shape must be set to resize camera")
        scale_x = new_shape[1] / self.image_shape[1]
        scale_y = new_shape[0] / self.image_shape[0]
        new_K = self.K.clone()
        new_K[0, :] *= scale_x
        new_K[1, :] *= scale_y
        return self.copy(K=new_K, image_shape=tuple(new_shape))

    def principal_point_centered(self):
        """Return a new camera with principal point at image center."""
        if self.image_shape is None:
            raise ValueError("image_shape must be set for principal_point_centered")
        new_K = self.K.clone()
        new_K[0, 2] = (self.image_shape[1] - 1) / 2
        new_K[1, 2] = (self.image_shape[0] - 1) / 2
        return self.copy(K=new_K)

    def image_hflipped(self):
        """Return a camera adjusted for a horizontally flipped image."""
        if self.image_shape is None:
            raise ValueError("image_shape must be set for image_hflipped")

        if self.has_nonfisheye_distortion():
            new_R, new_K, new_d = _transform_coeffs_for_hflip(
                self.R, self.K, self.d, self.image_shape)
            return self.copy(R=new_R, K=new_K, d=new_d)

        # Fisheye or no distortion
        new_R = self.R.clone()
        new_R[0] *= -1
        new_K = self.K.clone()
        new_K[0, 2] = (self.image_shape[1] - 1) - new_K[0, 2]
        new_K[0, 1] *= -1
        return self.copy(R=new_R, K=new_K)

    def image_rotated(self, angle, anchor=None):
        """Return a camera adjusted for a rotated image.

        Args:
            angle: Rotation angle in radians (counter-clockwise)
            anchor: Rotation center (x, y). If None, uses image center.
        """
        device, dtype = self.device, self.K.dtype

        if anchor is None:
            if self.image_shape is None:
                raise ValueError("image_shape must be set when anchor is None")
            anchor = torch.tensor(
                [(self.image_shape[1] - 1) / 2, (self.image_shape[0] - 1) / 2],
                device=device, dtype=dtype)
        else:
            anchor = torch.as_tensor(anchor, device=device, dtype=dtype)

        if self.has_nonfisheye_distortion():
            new_R, new_K, new_d = _transform_coeffs_for_rotation(
                self.R, self.K, self.d, angle, anchor)
            return self.copy(R=new_R, K=new_K, d=new_d)

        # Fisheye or no distortion
        angle = torch.as_tensor(angle, device=device, dtype=dtype)
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        rot_image = torch.stack([cos, -sin, sin, cos]).reshape(2, 2)

        new_K = self.K.clone()
        v = rot_image[1] @ new_K[:2, :2]
        v = v / torch.linalg.norm(v)
        rot_norm = torch.stack([v[1], -v[0], v[0], v[1]]).reshape(2, 2)

        new_K[:2, :2] = rot_image @ new_K[:2, :2] @ rot_norm.T
        new_K[:2, 2] = rot_image @ (new_K[:2, 2] - anchor) + anchor

        new_R = self.R.clone()
        new_R[:2] = rot_norm @ new_R[:2]

        return self.copy(R=new_R, K=new_K)

    def image_rot90(self, k=1):
        """Return a camera adjusted for a 90-degree rotated image.

        Args:
            k: Number of 90-degree rotations (counter-clockwise)
        """
        if self.image_shape is None:
            raise ValueError("image_shape must be set for image_rot90")

        k = k % 4
        if k == 0:
            return self.copy()
        elif k == 1:
            a = (self.image_shape[0] - 1) / 2
            new_shape = (self.image_shape[1], self.image_shape[0])
            return self.image_rotated(math.pi / 2, anchor=(a, a)).copy(
                image_shape=new_shape)
        elif k == 2:
            return self.image_rotated(math.pi)
        else:  # k == 3
            a = (self.image_shape[1] - 1) / 2
            new_shape = (self.image_shape[1], self.image_shape[0])
            return self.image_rotated(-math.pi / 2, anchor=(a, a)).copy(
                image_shape=new_shape)


# =================================================================
# Module-level functions
# =================================================================

def reproject_image(image, old_camera, new_camera, output_shape=None, **kwargs):
    """Reproject image between cameras.

    Args:
        image: (B, C, H, W) image tensor
        old_camera: source pt.Camera
        new_camera: target pt.Camera
        output_shape: (h, w), defaults to new_camera.image_shape
        **kwargs: passed to reprojection.reproject_image (antialias_factor, mode,
            padding_mode)

    Returns:
        (B, C, out_h, out_w) reprojected image
    """
    if output_shape is None:
        output_shape = new_camera.image_shape
        if output_shape is None:
            raise ValueError(
                "output_shape required: either pass it explicitly or set "
                "new_camera.image_shape")

    return reproj_module.reproject_image(
        image, old_camera.K, old_camera.R, old_camera.d,
        new_camera.K, new_camera.R, new_camera.d,
        output_shape, **kwargs)


def reproject_image_multi(images, old_cameras, new_cameras, output_shape=None, **kwargs):
    """Reproject a batch of images, each with its own camera pair.

    Uses batched grid generation for efficiency when distortion is shared
    across the batch. Falls back to sequential otherwise.

    Args:
        images: (B, C, H, W) image batch
        old_cameras: sequence of B Camera objects, or single Camera
        new_cameras: sequence of B Camera objects, or single Camera
        output_shape: (h, w), defaults to new_cameras[0].image_shape
        **kwargs: passed to reprojection.reproject_image_multi
            (antialias_factor, mode, padding_mode)

    Returns:
        (B, C, out_h, out_w) reprojected images
    """
    if output_shape is None:
        cam = new_cameras[0] if isinstance(new_cameras, (list, tuple)) else new_cameras
        output_shape = cam.image_shape
        if output_shape is None:
            raise ValueError(
                "output_shape required: either pass it explicitly or set "
                "new_camera.image_shape")

    return reproj_module.reproject_image_multi(
        images, old_cameras, new_cameras, output_shape, **kwargs)


def reproject_image_points(points, old_camera, new_camera):
    """Reproject 2D points between cameras.

    Args:
        points: (..., 2) pixel coordinates in old camera
        old_camera: source pt.Camera
        new_camera: target pt.Camera

    Returns:
        (..., 2) pixel coordinates in new camera
    """
    return reproj_module.reproject_image_points(
        points, old_camera.K, old_camera.R, old_camera.d,
        new_camera.K, new_camera.R, new_camera.d)


def reproject_depth_map(depth, old_camera, new_camera, output_shape=None, **kwargs):
    """Reproject depth map with z-correction.

    Args:
        depth: (B, 1, H, W) depth map
        old_camera: source pt.Camera
        new_camera: target pt.Camera
        output_shape: (h, w), defaults to new_camera.image_shape
        **kwargs: passed to reprojection.reproject_depth_map (antialias_factor, mode)

    Returns:
        (B, 1, out_h, out_w) reprojected depth map
    """
    if output_shape is None:
        output_shape = new_camera.image_shape
        if output_shape is None:
            raise ValueError(
                "output_shape required: either pass it explicitly or set "
                "new_camera.image_shape")

    return reproj_module.reproject_depth_map(
        depth, old_camera.K, old_camera.R, old_camera.d,
        new_camera.K, new_camera.R, new_camera.d,
        output_shape, **kwargs)


# =================================================================
# Helpers
# =================================================================

def _to_tensor(x, device=None, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x if device is None else x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32)).to(
            device=device or 'cpu', dtype=dtype)
    return torch.tensor(x, device=device or 'cpu', dtype=dtype)


def _unit_vec(v):
    """Normalize a vector along the last dimension."""
    return v / torch.linalg.norm(v, dim=-1, keepdim=True)


def _rodrigues(axis_angle):
    """Convert axis-angle vector to 3x3 rotation matrix (Rodrigues formula)."""
    angle = torch.linalg.norm(axis_angle)
    k = axis_angle / angle
    K = torch.stack([
        torch.zeros_like(k[0]), -k[2], k[1],
        k[2], torch.zeros_like(k[0]), -k[0],
        -k[1], k[0], torch.zeros_like(k[0])
    ]).reshape(3, 3)
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def _euler_xyz_to_matrix(a, b, c, device, dtype):
    """Build rotation matrix from extrinsic XYZ Euler angles: R = Rz(c) @ Ry(b) @ Rx(a)."""
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cc, sc = torch.cos(c), torch.sin(c)
    z = torch.zeros((), device=device, dtype=dtype)
    o = torch.ones((), device=device, dtype=dtype)
    Rx = torch.stack([o, z, z, z, ca, -sa, z, sa, ca]).reshape(3, 3)
    Ry = torch.stack([cb, z, sb, z, o, z, -sb, z, cb]).reshape(3, 3)
    Rz = torch.stack([cc, -sc, z, sc, cc, z, z, z, o]).reshape(3, 3)
    return Rz @ Ry @ Rx


def _matrix_to_euler_zxy(R):
    """Extract extrinsic ZXY Euler angles from rotation matrix.

    R = Ry(y) @ Rx(x) @ Rz(z), returns (z, x, y).
    """
    # R[1,2] = -sin(x)
    x = -torch.asin(R[1, 2])
    # R[0,2] = cos(x)*sin(y), R[2,2] = cos(x)*cos(y)
    y = torch.atan2(R[0, 2], R[2, 2])
    # R[1,0] = cos(x)*sin(z), R[1,1] = cos(x)*cos(z)
    z = torch.atan2(R[1, 0], R[1, 1])
    return z, x, y


def _euler_xy_to_matrix(a, b, device, dtype):
    """Build rotation matrix from extrinsic XY Euler angles: R = Ry(b) @ Rx(a)."""
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    z = torch.zeros((), device=device, dtype=dtype)
    o = torch.ones((), device=device, dtype=dtype)
    Rx = torch.stack([o, z, z, z, ca, -sa, z, sa, ca]).reshape(3, 3)
    Ry = torch.stack([cb, z, sb, z, o, z, -sb, z, cb]).reshape(3, 3)
    return Ry @ Rx


def _transform_coeffs_for_hflip(R, K, d, imshape):
    """Transform R, K, d for horizontal image flip (BrownConrady distortion)."""
    new_R = R.clone()
    new_R[0] *= -1

    new_K = K.clone()
    new_K[0, 2] = (imshape[1] - 1) - new_K[0, 2]
    new_K[0, 1] *= -1

    new_d = d.clone()
    new_d[3] *= -1  # p2
    n = new_d.shape[0]
    if n > 8:
        new_d[8] *= -1  # s1
        new_d[9] *= -1  # s2
    if n >= 14:
        new_d[13] *= -1  # tau_y

    return new_R, new_K, new_d


def _transform_coeffs_for_rotation(R, K, d, angle, anchor):
    """Transform R, K, d for image rotation (BrownConrady distortion)."""
    device, dtype = K.device, K.dtype
    angle = torch.as_tensor(angle, device=device, dtype=dtype)
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    rot_image = torch.stack([cos, -sin, sin, cos]).reshape(2, 2)

    # Compute rot_normalized: rotation in normalized coords that avoids K[1,0] skew
    new_K = K.clone()
    v = rot_image[1] @ new_K[:2, :2]
    v = v / torch.linalg.norm(v)
    rot_normalized = torch.stack([v[1], -v[0], v[0], v[1]]).reshape(2, 2)

    new_K[:2, :2] = rot_image @ new_K[:2, :2] @ rot_normalized.T
    new_K[:2, 2] = rot_image @ (new_K[:2, 2] - anchor) + anchor

    new_R = R.clone()
    new_R[:2] = rot_normalized @ new_R[:2]

    new_d = d.clone()
    n = new_d.shape[0]

    # Handle tilt (tau_x, tau_y) if present
    has_tilt = n >= 14 and (new_d[12] != 0 or new_d[13] != 0)
    if has_tilt:
        angle_normalized = torch.atan2(v[0], v[1])
        tau_x_old = d[12]
        tau_y_old = d[13]

        # Euler reorder: xyz -> zxy
        R_euler = _euler_xyz_to_matrix(tau_x_old, tau_y_old, angle_normalized, device, dtype)
        angle_coeffs, new_tau_x, new_tau_y = _matrix_to_euler_zxy(R_euler)
        new_d[12] = new_tau_x
        new_d[13] = new_tau_y

        # rot_coeffs from the extracted z-angle
        cos_c = torch.cos(angle_coeffs)
        sin_c = torch.sin(angle_coeffs)
        rot_coeffs = torch.stack([cos_c, -sin_c, sin_c, cos_c]).reshape(2, 2)

        # Tilt homography correction
        tilt_rot = _euler_xy_to_matrix(tau_x_old, tau_y_old, device, dtype)
        tilt_homography = torch.stack([
            tilt_rot[2, 2], torch.zeros((), device=device, dtype=dtype), -tilt_rot[0, 2],
            torch.zeros((), device=device, dtype=dtype), tilt_rot[2, 2], -tilt_rot[1, 2],
            torch.zeros((), device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
            torch.ones((), device=device, dtype=dtype)
        ]).reshape(3, 3)

        rot_norm_3x3 = torch.eye(3, device=device, dtype=dtype)
        rot_norm_3x3[:2, :2] = rot_normalized
        tilt_homography_rotated = rot_norm_3x3 @ tilt_homography @ rot_norm_3x3.T

        tilt_rot_new = _euler_xy_to_matrix(new_tau_x, new_tau_y, device, dtype)
        tilt_scale = 1 / tilt_rot_new[2, 2]
        tilt_homography_new_inv = torch.stack([
            tilt_scale, torch.zeros((), device=device, dtype=dtype),
            tilt_rot_new[0, 2] * tilt_scale,
            torch.zeros((), device=device, dtype=dtype), tilt_scale,
            tilt_rot_new[1, 2] * tilt_scale,
            torch.zeros((), device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
            torch.ones((), device=device, dtype=dtype)
        ]).reshape(3, 3)

        new_K = new_K @ tilt_homography_rotated @ tilt_homography_new_inv
    else:
        rot_coeffs = rot_normalized

    # Rotate tangential (p1, p2) and thin prism (s1-s4)
    p1p2 = torch.stack([new_d[3], new_d[2]])
    p1p2_new = rot_coeffs @ p1p2
    new_d[3] = p1p2_new[0]
    new_d[2] = p1p2_new[1]

    if n > 8:
        s_block = torch.stack([
            torch.stack([new_d[8], new_d[9]]),
            torch.stack([new_d[10], new_d[11]])
        ])
        s_block_new = rot_coeffs @ s_block
        new_d[8] = s_block_new[0, 0]
        new_d[9] = s_block_new[0, 1]
        new_d[10] = s_block_new[1, 0]
        new_d[11] = s_block_new[1, 1]

    return new_R, new_K, new_d
