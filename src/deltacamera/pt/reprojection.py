
import torch
import torch.nn.functional as F

from . import maps


def reproject_image(
    image, old_K, old_R, old_d, new_K, new_R, new_d, output_shape,
    antialias_factor=1, mode='bilinear', padding_mode='zeros',
    precomp_undist_maps=True,
):
    """Warp image from old camera to new camera.

    Args:
        image: (B, C, H, W) input image tensor
        old_K: (3, 3) old camera intrinsics
        old_R: (3, 3) old camera rotation
        old_d: (C,) old distortion coefficients, or None
        new_K: (3, 3) new camera intrinsics
        new_R: (3, 3) new camera rotation
        new_d: (C,) new distortion coefficients, or None
        output_shape: (out_h, out_w)
        antialias_factor: supersample factor for antialiasing (1 = no antialiasing)
        mode: interpolation mode ('bilinear' or 'bicubic')
        padding_mode: 'zeros', 'border', or 'reflection'
        precomp_undist_maps: if True, use cached undistortion lookup table for the
            new camera instead of the iterative solver. Useful when only rotation
            changes between calls.

    Returns:
        (B, C, out_h, out_w) warped image
    """
    B = image.shape[0]
    in_h, in_w = image.shape[2], image.shape[3]
    out_h, out_w = output_shape

    if antialias_factor > 1:
        a = antialias_factor
        aa_h, aa_w = out_h * a, out_w * a
        # Scale new intrinsics for higher resolution
        scale_mat = torch.eye(3, device=new_K.device, dtype=new_K.dtype)
        scale_mat[0, 0] = a
        scale_mat[1, 1] = a
        # Shift to maintain pixel centers (center_subpixels=True equivalent)
        scale_mat[0, 2] = (a - 1) / 2.0
        scale_mat[1, 2] = (a - 1) / 2.0
        new_K_aa = scale_mat @ new_K

        grid = maps.make_remap_grid(
            old_K, old_R, old_d, new_K_aa, new_R, new_d,
            aa_h, aa_w, old_imshape=(in_h, in_w), precomp_undist_maps=precomp_undist_maps)
    else:
        grid = maps.make_remap_grid(
            old_K, old_R, old_d, new_K, new_R, new_d,
            out_h, out_w, old_imshape=(in_h, in_w), precomp_undist_maps=precomp_undist_maps)

    # Expand grid to batch
    grid = grid.expand(B, -1, -1, -1)

    # grid_sample's behavior for NaN grid coordinates is undefined (bilinear+zeros
    # yields NaN, nearest yields 0, contradicting the docs); replace NaN with a large
    # value so those samples are consistently treated as out-of-bounds.
    grid = torch.where(torch.isnan(grid), 1e6, grid)

    result = F.grid_sample(image, grid, mode=mode, padding_mode=padding_mode,
                           align_corners=True)

    if antialias_factor > 1:
        result = F.avg_pool2d(result, kernel_size=antialias_factor,
                              stride=antialias_factor)

    return result


def reproject_image_multi(
    images, old_cameras, new_cameras, output_shape,
    antialias_factor=1, mode='bilinear', padding_mode='zeros',
    precomp_undist_maps=True,
):
    """Warp a batch of images, each with its own camera pair.

    Uses batched grid generation when at least one side's distortion is
    shared across the batch. The shared side is computed once; the other
    side loops per element. Falls back to fully sequential only when
    neither side's distortion is shared.

    Args:
        images: (B, C, H, W) input image batch
        old_cameras: sequence of B Camera objects, or single Camera
        new_cameras: sequence of B Camera objects, or single Camera
        output_shape: (out_h, out_w)
        antialias_factor: supersample factor (1 = none)
        mode: interpolation mode ('bilinear' or 'bicubic')
        padding_mode: 'zeros', 'border', or 'reflection'
        precomp_undist_maps: if True, use cached undistortion lookup tables
            in the sequential fallback path (passed to make_remap_grid)

    Returns:
        (B, C, out_h, out_w) warped images
    """
    B = images.shape[0]
    in_h, in_w = images.shape[2], images.shape[3]
    out_h, out_w = output_shape
    device = images.device

    # The batch size is determined by both the images and the camera lists
    # (a single image may be broadcast over multiple camera pairs)
    n_batch = max(
        B,
        len(old_cameras) if isinstance(old_cameras, (list, tuple)) else 1,
        len(new_cameras) if isinstance(new_cameras, (list, tuple)) else 1,
    )

    # Broadcast single camera to all batch elements
    if not isinstance(old_cameras, (list, tuple)):
        old_cameras = [old_cameras] * n_batch
    if not isinstance(new_cameras, (list, tuple)):
        new_cameras = [new_cameras] * n_batch

    old_ds = [c.d for c in old_cameras]
    new_ds = [c.d for c in new_cameras]
    old_d_shared = _check_shared_d(old_ds)
    new_d_shared = _check_shared_d(new_ds)

    if old_d_shared is _NOT_SHARED and new_d_shared is _NOT_SHARED:
        # Sequential fallback: neither side's distortion is shared
        if antialias_factor > 1:
            a = antialias_factor
            aa_h, aa_w = out_h * a, out_w * a
            scale_mat = torch.eye(3, device=device, dtype=old_cameras[0].K.dtype)
            scale_mat[0, 0] = a
            scale_mat[1, 1] = a
            scale_mat[0, 2] = (a - 1) / 2.0
            scale_mat[1, 2] = (a - 1) / 2.0
            grids = []
            for i in range(n_batch):
                g = maps.make_remap_grid(
                    old_cameras[i].K, old_cameras[i].R, old_cameras[i].d,
                    scale_mat @ new_cameras[i].K, new_cameras[i].R,
                    new_cameras[i].d,
                    aa_h, aa_w, old_imshape=(in_h, in_w),
                    precomp_undist_maps=precomp_undist_maps)
                grids.append(g)
        else:
            grids = []
            for i in range(n_batch):
                g = maps.make_remap_grid(
                    old_cameras[i].K, old_cameras[i].R, old_cameras[i].d,
                    new_cameras[i].K, new_cameras[i].R, new_cameras[i].d,
                    out_h, out_w, old_imshape=(in_h, in_w),
                    precomp_undist_maps=precomp_undist_maps)
                grids.append(g)
        grid = torch.cat(grids, dim=0)
    else:
        # Batched path: at least one side's distortion is shared
        old_Ks = _maybe_stack([c.K for c in old_cameras])
        old_Rs = torch.stack([c.R for c in old_cameras])
        new_Ks = _maybe_stack([c.K for c in new_cameras])
        new_Rs = torch.stack([c.R for c in new_cameras])

        old_d_arg = old_d_shared if old_d_shared is not _NOT_SHARED else old_ds
        new_d_arg = new_d_shared if new_d_shared is not _NOT_SHARED else new_ds

        if antialias_factor > 1:
            a = antialias_factor
            aa_h, aa_w = out_h * a, out_w * a
            scale_mat = torch.eye(3, device=device, dtype=new_Ks.dtype)
            scale_mat[0, 0] = a
            scale_mat[1, 1] = a
            scale_mat[0, 2] = (a - 1) / 2.0
            scale_mat[1, 2] = (a - 1) / 2.0
            new_Ks = scale_mat @ new_Ks
            grid = maps.make_remap_grid_batched(
                old_Ks, old_Rs, old_d_arg,
                new_Ks, new_Rs, new_d_arg,
                aa_h, aa_w, old_imshape=(in_h, in_w))
        else:
            grid = maps.make_remap_grid_batched(
                old_Ks, old_Rs, old_d_arg,
                new_Ks, new_Rs, new_d_arg,
                out_h, out_w, old_imshape=(in_h, in_w))

    grid = torch.where(torch.isnan(grid), 1e6, grid)
    if images.shape[0] == 1 and grid.shape[0] > 1:
        images = images.expand(grid.shape[0], -1, -1, -1)
    result = F.grid_sample(
        images, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if antialias_factor > 1:
        result = F.avg_pool2d(
            result, kernel_size=antialias_factor, stride=antialias_factor)

    return result


_NOT_SHARED = object()


def _check_shared_d(ds):
    """Return the shared d tensor if all are identical, else _NOT_SHARED."""
    if all(d is None for d in ds):
        return None
    if any(d is None for d in ds):
        return _NOT_SHARED
    ref = ds[0]
    for d in ds[1:]:
        if d.shape != ref.shape or not torch.equal(d, ref):
            return _NOT_SHARED
    return ref


def _maybe_stack(tensors):
    """Stack tensors, but return single (3,3) if all are identical."""
    ref = tensors[0]
    if all(t.data_ptr() == ref.data_ptr() for t in tensors[1:]):
        return ref
    if all(torch.equal(t, ref) for t in tensors[1:]):
        return ref
    return torch.stack(tensors)


def reproject_image_points(points, old_K, old_R, old_d, new_K, new_R, new_d):
    """Reproject 2D image points from old camera to new camera.

    Note: point reprojection goes the OPPOSITE direction from image warping.
    Image warping maps new→old; point reprojection maps old→new.

    Args:
        points: (..., 2) pixel coordinates in old camera
        old_K, old_R, old_d: old camera parameters
        new_K, new_R, new_d: new camera parameters

    Returns:
        (..., 2) pixel coordinates in new camera
    """
    # Swap cameras: old→new means undo old, rotate to new, apply new
    return maps.make_remap_from_points(
        new_K, new_R, new_d, old_K, old_R, old_d, points)


def reproject_depth_map(
    depth, old_K, old_R, old_d, new_K, new_R, new_d, output_shape,
    antialias_factor=1, mode='nearest',
):
    """Reproject depth map with z-correction.

    When ``antialias_factor`` > 1, the depth is rendered at higher resolution
    and block-downsampled with nanmedian (NaN-safe, edge-preserving).

    Args:
        depth: (B, 1, H, W) depth map
        old_K, old_R, old_d: old camera parameters
        new_K, new_R, new_d: new camera parameters
        output_shape: (out_h, out_w)
        antialias_factor: supersample factor
        mode: interpolation ('nearest' recommended for depth)

    Returns:
        (B, 1, out_h, out_w) reprojected depth map
    """
    out_h, out_w = output_shape
    a = antialias_factor

    if a > 1:
        hr_h, hr_w = out_h * a, out_w * a
        scale_mat = torch.eye(3, device=new_K.device, dtype=new_K.dtype)
        scale_mat[0, 0] = a
        scale_mat[1, 1] = a
        scale_mat[0, 2] = (a - 1) / 2.0
        scale_mat[1, 2] = (a - 1) / 2.0
        hr_new_K = scale_mat @ new_K
    else:
        hr_h, hr_w = out_h, out_w
        hr_new_K = new_K

    # Warp at high resolution with an analytic out-of-bounds mask. grid_sample's zero
    # padding cannot mark invalid pixels for depth: a 0 sentinel destroys legitimate
    # zero depths, and bilinear samples that partially leave the image blend with the
    # padding into finite garbage. Instead, compute the grid in raw pixel coordinates
    # and mark exactly the samples whose interpolation taps fall outside the image
    # (matching cv2.remap's borderValue=NaN semantics).
    in_h, in_w = depth.shape[2], depth.shape[3]
    grid_px = maps.make_remap_grid(
        old_K, old_R, old_d, hr_new_K, new_R, new_d, hr_h, hr_w, old_imshape=None)
    x, y = grid_px[..., 0], grid_px[..., 1]
    if mode == 'nearest':
        valid = (x > -0.5) & (x < in_w - 0.5) & (y > -0.5) & (y < in_h - 0.5)
    else:
        # bilinear: invalid if any of the 2x2 taps falls outside the image
        valid = (x >= 0) & (x <= in_w - 1) & (y >= 0) & (y <= in_h - 1)
    # NaN grid coords (invalid distortion region) fail the comparisons -> invalid;
    # replace them before sampling since grid_sample's NaN handling is undefined.
    grid = maps._normalize_grid(grid_px, in_h, in_w)
    grid = torch.nan_to_num(grid, nan=2.0)
    warped = F.grid_sample(
        depth, grid.expand(depth.shape[0], -1, -1, -1),
        mode=mode, padding_mode='zeros', align_corners=True)
    warped = torch.where(valid.unsqueeze(1), warped, float('nan'))

    # Compute z-correction factors at the high-res size
    z_factors = maps.make_z_factors(old_R, hr_new_K, new_R, new_d, hr_h, hr_w)

    # Apply: depth_new = depth_old / z_factor
    z_factors = z_factors.unsqueeze(0).unsqueeze(0)  # (1, 1, hr_H, hr_W)
    safe_z = torch.where(z_factors > 0, z_factors, torch.ones_like(z_factors))
    result = warped / safe_z
    result = torch.where(z_factors > 0, result, float('nan'))

    if a > 1:
        # Block nanmedian downsample (edge-preserving, NaN-safe).
        # The numpy version also applies a bilateral filter before this step;
        # a future improvement could add an edge-preserving smooth here too
        # (e.g. via kornia.filters.bilateral_blur).
        result = _block_nanmedian(result, a)

    return result


def _block_nanmedian(x, block_size):
    """Block-reduce a (B, C, H, W) tensor by nanmedian with block_size x block_size blocks."""
    B, C, H, W = x.shape
    a = block_size
    blocks = x.unfold(2, a, a).unfold(3, a, a)  # (B, C, H//a, W//a, a, a)
    blocks = blocks.reshape(B, C, H // a, W // a, a * a)
    return torch.nanmedian(blocks, dim=-1).values
