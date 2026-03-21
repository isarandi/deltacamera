import torch


def world_to_image(points, K, R, t):
    """Project 3D world points to 2D pixel coordinates (no distortion).

    Args:
        points: (..., 3) world coordinates
        K: (..., 3, 3) intrinsic matrix
        R: (..., 3, 3) rotation matrix (world-to-camera)
        t: (..., 3) camera position in world coordinates

    Returns:
        (..., 2) pixel coordinates. NaN where z <= 0.
    """
    cam_pts = world_to_camera(points, R, t)
    norm_pts = project(cam_pts)
    return apply_intrinsics(norm_pts, K)


def world_to_camera(points, R, t):
    """Transform world coordinates to camera coordinates: R @ (points - t).

    Args:
        points: (..., 3) world coordinates
        R: (..., 3, 3) rotation matrix (world-to-camera)
        t: (..., 3) camera position in world coordinates

    Returns:
        (..., 3) camera coordinates
    """
    return torch.einsum('...ij,...j->...i', R, points - t)


def camera_to_world(points, R, t):
    """Transform camera coordinates to world coordinates: R^T @ points + t.

    Args:
        points: (..., 3) camera coordinates
        R: (..., 3, 3) rotation matrix (world-to-camera)
        t: (..., 3) camera position in world coordinates

    Returns:
        (..., 3) world coordinates
    """
    return torch.einsum('...ji,...j->...i', R, points) + t


def world_to_undist(points, R, t):
    """Project world coordinates to normalized undistorted 2D coordinates.

    Args:
        points: (..., 3) world coordinates
        R: (..., 3, 3) rotation matrix
        t: (..., 3) camera position

    Returns:
        (..., 2) normalized undistorted coordinates. NaN where z <= 0.
    """
    cam_pts = world_to_camera(points, R, t)
    return project(cam_pts)


def project(points):
    """Perspective projection: (x, y, z) -> (x/z, y/z).

    Args:
        points: (..., 3)

    Returns:
        (..., 2). NaN where z <= 0.
    """
    z = points[..., 2:3]
    safe_z = torch.where(z > 0, z, torch.ones_like(z))
    result = points[..., :2] / safe_z
    return torch.where(z > 0, result, float('nan'))


def apply_intrinsics(points, K):
    """Apply intrinsic matrix to normalized 2D coordinates.

    Args:
        points: (..., 2) normalized image coordinates
        K: (..., 3, 3) intrinsic matrix (upper-triangular)

    Returns:
        (..., 2) pixel coordinates
    """
    x = points[..., 0]
    y = points[..., 1]
    fx = K[..., 0, 0]
    skew = K[..., 0, 1]
    cx = K[..., 0, 2]
    fy = K[..., 1, 1]
    cy = K[..., 1, 2]
    px = fx * x + skew * y + cx
    py = fy * y + cy
    return torch.stack([px, py], dim=-1)


def undo_intrinsics(points, K):
    """Remove intrinsic matrix from pixel coordinates.

    Args:
        points: (..., 2) pixel coordinates
        K: (..., 3, 3) intrinsic matrix (upper-triangular)

    Returns:
        (..., 2) normalized image coordinates
    """
    fx = K[..., 0, 0]
    skew = K[..., 0, 1]
    cx = K[..., 0, 2]
    fy = K[..., 1, 1]
    cy = K[..., 1, 2]
    y_n = (points[..., 1] - cy) / fy
    x_n = (points[..., 0] - cx - skew * y_n) / fx
    return torch.stack([x_n, y_n], dim=-1)


def backproject(points, K, depth=1.0):
    """Backproject 2D pixel coordinates to 3D camera coordinates.

    Args:
        points: (..., 2) pixel coordinates
        K: (..., 3, 3) intrinsic matrix
        depth: scalar, or tensor broadcastable to (...,)

    Returns:
        (..., 3) camera coordinates
    """
    norm_pts = undo_intrinsics(points, K)
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth, dtype=norm_pts.dtype, device=norm_pts.device)
    depth = depth.unsqueeze(-1) if depth.dim() > 0 and depth.shape != () else depth
    xy = norm_pts * depth
    z = depth * torch.ones_like(norm_pts[..., :1]) if depth.dim() == 0 else depth
    if z.shape[-1:] != (1,):
        z = z.unsqueeze(-1)
    return torch.cat([xy, z.expand_as(xy[..., :1])], dim=-1)


def transform_perspective(points, H):
    """Apply a 3x3 homography to 2D points with perspective division.

    Args:
        points: (..., 2)
        H: (..., 3, 3)

    Returns:
        (..., 2). NaN where z <= 0.
    """
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat([points, ones], dim=-1)
    transformed = torch.einsum('...ij,...j->...i', H, homogeneous)
    z = transformed[..., 2:3]
    safe_z = torch.where(z > 0, z, torch.ones_like(z))
    result = transformed[..., :2] / safe_z
    return torch.where(z > 0, result, float('nan'))


def make_pixel_grid(h, w, device=None, dtype=torch.float32):
    """Generate a grid of pixel coordinates (x, y) for an (h, w) image.

    Args:
        h: image height
        w: image width
        device: torch device
        dtype: tensor dtype

    Returns:
        (h, w, 2) tensor of (x, y) pixel coordinates
    """
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing='ij',
    )
    return torch.stack([x, y], dim=-1)
