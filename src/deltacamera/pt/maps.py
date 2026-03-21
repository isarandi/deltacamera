import torch

from . import coordframes
from . import distortion as dist_module
from . import validity


def make_remap_grid(
    old_K, old_R, old_d, new_K, new_R, new_d, h, w, old_imshape=None,
    precomp_undist_maps=False,
):
    """Generate a remap grid for image warping from old camera to new camera.

    Each output pixel in the new camera is mapped to its source coordinate in the
    old camera. The pipeline: undo new intrinsics → undistort new → rotate → distort
    old → apply old intrinsics.

    Args:
        old_K: (3, 3) old camera intrinsics
        old_R: (3, 3) old camera rotation
        old_d: (C,) old camera distortion coefficients, or None
        new_K: (3, 3) new camera intrinsics
        new_R: (3, 3) new camera rotation
        new_d: (C,) new camera distortion coefficients, or None
        h, w: output image dimensions
        old_imshape: (old_h, old_w) — if provided, normalizes grid to [-1, 1]
            for grid_sample. If None, returns raw pixel coordinates.
        precomp_undist_maps: if True, use a cached undistortion lookup table for
            the new camera instead of the iterative solver. Faster when only
            rotation changes between calls (LUT is cached by coefficient values).
            Forward distortion (old camera) is always computed directly (cheap
            polynomial). Slight accuracy loss (~0.7px max at validity boundary,
            <0.001px typical).

    Returns:
        (1, h, w, 2) grid tensor. If old_imshape given, normalized for grid_sample.
    """
    if precomp_undist_maps:
        return make_remap_grid_from_precomputed(
            new_K, new_R, _get_cached_undistort(new_d),
            old_K, old_R, old_d,
            h, w, old_imshape=old_imshape)

    # Cache lookups outside the compilable pipeline (like numba's outer make())
    old_d_padded, old_valid = _prepare_distortion(old_d)
    new_d_padded, new_valid = _prepare_distortion(new_d)

    grid = coordframes.make_pixel_grid(h, w, device=old_K.device, dtype=old_K.dtype)
    result = _remap_pipeline(
        grid, old_K, old_R, old_d_padded, old_valid,
        new_K, new_R, new_d_padded, new_valid)

    if old_imshape is not None:
        result = _normalize_grid(result, old_imshape[0], old_imshape[1])

    return result.unsqueeze(0)


def make_remap_from_points(old_K, old_R, old_d, new_K, new_R, new_d, points):
    """Apply the remap pipeline to arbitrary points (not a regular grid).

    Args:
        points: (..., 2) pixel coordinates in the new camera
        Other args: same as make_remap_grid

    Returns:
        (..., 2) source pixel coordinates in the old camera
    """
    old_d_padded, old_valid = _prepare_distortion(old_d)
    new_d_padded, new_valid = _prepare_distortion(new_d)
    return _remap_pipeline(
        points, old_K, old_R, old_d_padded, old_valid,
        new_K, new_R, new_d_padded, new_valid)


# =============================================================================
# Precomputed distortion lookup tables (like CPU numba implementation)
# =============================================================================

def precompute_undistort(d, fov_degrees_max=120, res_bc=1024, res_fisheye=2048):
    """Precompute an undistortion lookup table, resolution-independent.

    For Brown-Conrady: a 2D grid storing undistorted coords + Jacobian for
    nearest-neighbor lookup with linear correction.
    For fisheye: a 1D table of radial scale factors.

    Args:
        d: (C,) distortion coefficients, or None
        fov_degrees_max: max FOV covered by BC lookup table (degrees)
        res_bc: resolution of BC lookup grid (res x res)
        res_fisheye: resolution of fisheye 1D lookup table

    Returns:
        Precomputed dict to pass to make_remap_grid_from_precomputed, or None.
    """
    if d is None:
        return None
    if d.shape[-1] == 4:
        return _precompute_undistort_fisheye(d, res_fisheye)
    else:
        return _precompute_undistort_bc(d, fov_degrees_max, res_bc)



def make_remap_grid_from_precomputed(
    new_K, new_R, new_undist_precomp,
    old_K, old_R, old_d,
    h, w, old_imshape=None,
):
    """Generate a remap grid using a precomputed undistortion lookup table.

    Much faster than make_remap_grid when the new camera's distortion is fixed
    and only the rotation changes between frames. Forward distortion (old camera)
    is computed directly (cheap polynomial).

    Args:
        new_K: (3, 3) new camera intrinsics
        new_R: (3, 3) new camera rotation
        new_undist_precomp: from precompute_undistort(new_d), or None
        old_K: (3, 3) old camera intrinsics
        old_R: (3, 3) old camera rotation
        old_d: (C,) old camera distortion coefficients, or None
        h, w: output image dimensions
        old_imshape: (old_h, old_w) for grid_sample normalization

    Returns:
        (1, h, w, 2) grid tensor
    """
    old_d_padded, old_valid = _prepare_distortion(old_d)

    grid = coordframes.make_pixel_grid(h, w, device=old_K.device, dtype=old_K.dtype)

    # Step 1: undo new camera intrinsics
    pn = coordframes.undo_intrinsics(grid, new_K)

    # Step 2: undistort new camera (precomputed lookup)
    pn = _apply_undistort_precomp(pn, new_undist_precomp)

    # Step 3: rotate from new camera to old camera
    R_rel = old_R @ new_R.T
    ones = torch.ones_like(pn[..., :1])
    pn_3d = torch.cat([pn, ones], dim=-1)
    rotated = torch.einsum('ij,...j->...i', R_rel, pn_3d)
    pn = coordframes.project(rotated)

    # Step 4: distort old camera (direct polynomial)
    pn = _distort(pn, old_d_padded, old_valid)

    # Step 5: apply old camera intrinsics
    p_old = coordframes.apply_intrinsics(pn, old_K)

    if old_imshape is not None:
        p_old = _normalize_grid(p_old, old_imshape[0], old_imshape[1])

    return p_old.unsqueeze(0)


def make_remap_grid_batched(
    old_Ks, old_Rs, old_d, new_Ks, new_Rs, new_d, h, w, old_imshape=None,
):
    """Batched remap grid generation for multiple camera pairs.

    Efficient when at least one side's distortion coefficients are shared
    across the batch. The shared side's distortion is computed once; the
    per-element side is looped. K and R can always vary per batch element.

    Args:
        old_Ks: (B, 3, 3) or (3, 3) old camera intrinsics
        old_Rs: (B, 3, 3) old camera rotations
        old_d: shared (C,) tensor or None, or list of B tensors/Nones
        new_Ks: (B, 3, 3) or (3, 3) new camera intrinsics
        new_Rs: (B, 3, 3) new camera rotations
        new_d: shared (C,) tensor or None, or list of B tensors/Nones
        h, w: output image dimensions
        old_imshape: (old_h, old_w) for grid_sample normalization

    Returns:
        (B, h, w, 2) grid tensor
    """
    B = old_Rs.shape[0]
    device = old_Rs.device
    dtype = old_Rs.dtype

    old_d_per_elem = isinstance(old_d, list)
    new_d_per_elem = isinstance(new_d, list)

    if old_d_per_elem:
        old_prepared = [_prepare_distortion(d) for d in old_d]
    else:
        old_d_padded, old_valid = _prepare_distortion(old_d)

    if new_d_per_elem:
        new_prepared = [_prepare_distortion(d) for d in new_d]
    else:
        new_d_padded, new_valid = _prepare_distortion(new_d)

    grid = coordframes.make_pixel_grid(h, w, device=device, dtype=dtype)

    # Step 1: undo new camera intrinsics
    if new_Ks.dim() == 2:
        pn = coordframes.undo_intrinsics(grid, new_Ks)  # (h, w, 2)
    else:
        pn = _batched_undo_intrinsics(grid, new_Ks)  # (B, h, w, 2)

    # Step 2: undistort new camera
    if new_d_per_elem:
        if pn.dim() == 3:
            pn = pn.unsqueeze(0).expand(B, -1, -1, -1)
        pn = torch.stack([
            _undistort(pn[i], *new_prepared[i]) for i in range(B)])
    else:
        pn = _undistort(pn, new_d_padded, new_valid)

    # Step 3: rotate from new camera to old camera
    R_rels = old_Rs @ new_Rs.transpose(-2, -1)  # (B, 3, 3)
    ones = torch.ones(*pn.shape[:-1], 1, device=device, dtype=dtype)
    pn_3d = torch.cat([pn, ones], dim=-1)
    if pn_3d.dim() == 3:
        rotated = torch.einsum('bij,hwj->bhwi', R_rels, pn_3d)
    else:
        rotated = torch.einsum('bij,bhwj->bhwi', R_rels, pn_3d)
    pn = coordframes.project(rotated)  # (B, h, w, 2)

    # Step 4: distort old camera
    if old_d_per_elem:
        pn = torch.stack([
            _distort(pn[i], *old_prepared[i]) for i in range(B)])
    else:
        pn = _distort(pn, old_d_padded, old_valid)

    # Step 5: apply old camera intrinsics
    if old_Ks.dim() == 2:
        p_old = coordframes.apply_intrinsics(pn, old_Ks)
    else:
        p_old = _batched_apply_intrinsics(pn, old_Ks)

    if old_imshape is not None:
        p_old = _normalize_grid(p_old, old_imshape[0], old_imshape[1])

    return p_old


def make_z_factors(old_R, new_K, new_R, new_d, h, w):
    """Compute per-pixel z-correction factors for depth reprojection.

    The z-factor is the z-component of the ray direction when rotated from the
    new camera frame to the old camera frame.

    Args:
        old_R: (3, 3) old camera rotation
        new_K: (3, 3) new camera intrinsics
        new_R: (3, 3) new camera rotation
        new_d: (C,) new camera distortion, or None
        h, w: output dimensions

    Returns:
        (h, w) z-factor tensor
    """
    new_d_padded, new_valid = _prepare_distortion(new_d)

    grid = coordframes.make_pixel_grid(h, w, device=old_R.device, dtype=old_R.dtype)
    pn = coordframes.undo_intrinsics(grid, new_K)
    pn = _undistort(pn, new_d_padded, new_valid)

    R_rel = old_R @ new_R.T
    R_row2 = R_rel[2, :]
    z_factor = pn[..., 0] * R_row2[0] + pn[..., 1] * R_row2[1] + R_row2[2]
    return z_factor


# =============================================================================
# Caching (like CPU numba's @lru_cache keyed on d.tobytes())
# =============================================================================

_MAX_CACHE_SIZE = 8192
_validity_cache = {}
_undistort_precomp_cache = {}


def _d_cache_key(d):
    if d is None:
        return None
    return (d.detach().cpu().numpy().tobytes(), str(d.device))


def _get_cached_undistort(d):
    """Get or compute cached undistortion LUT for given distortion coefficients."""
    key = _d_cache_key(d)
    if key not in _undistort_precomp_cache:
        if len(_undistort_precomp_cache) >= _MAX_CACHE_SIZE:
            _undistort_precomp_cache.pop(next(iter(_undistort_precomp_cache)))
        _undistort_precomp_cache[key] = precompute_undistort(d)
    return _undistort_precomp_cache[key]



@torch.compiler.disable
def _prepare_distortion(d) -> tuple:
    """Pad coefficients and look up cached validity. Called outside compiled region.

    This function is excluded from torch.compile tracing because it performs
    cache lookups (dict access with .tobytes() keys). The compilable inner
    pipeline (_remap_pipeline) receives the results as arguments.

    Returns:
        (d_padded, valid_info) where valid_info depends on distortion type.
        For None: (None, None)
        For fisheye: (d, (ru, rd))
        For BC: (d_padded_14, ((ru, tu), (rd, td)))
    """
    if d is None:
        return None, None
    if d.shape[-1] == 4:
        return d, _fisheye_valid_r_max_cached(d)
    else:
        d_padded = dist_module._pad_distortion_coeffs(d, 14)
        return d_padded, _bc_valid_region_cached(d_padded)


def _fisheye_valid_r_max_cached(d):
    key = ('fisheye', _d_cache_key(d))
    if key not in _validity_cache:
        if len(_validity_cache) >= _MAX_CACHE_SIZE:
            _validity_cache.pop(next(iter(_validity_cache)))
        _validity_cache[key] = validity.fisheye_valid_r_max(d)
    return _validity_cache[key]


def _bc_valid_region_cached(d):
    key = ('bc', _d_cache_key(d))
    if key not in _validity_cache:
        if len(_validity_cache) >= _MAX_CACHE_SIZE:
            _validity_cache.pop(next(iter(_validity_cache)))
        _validity_cache[key] = validity.brown_conrady_valid_region(d)
    return _validity_cache[key]


# =============================================================================
# Internal pipeline (compilable — no cache lookups, receives validity as args)
# =============================================================================

def _remap_pipeline(points, old_K, old_R, old_d, old_valid,
                    new_K, new_R, new_d, new_valid):
    """Core remap pipeline: new camera pixels → old camera pixels.

    Pipeline: undo_intrinsics(new) → undistort(new) → rotate → distort(old) →
    apply_intrinsics(old).

    All validity info is pre-fetched; this function is torch.compile-friendly.
    """
    # Step 1: undo new camera intrinsics
    pn = coordframes.undo_intrinsics(points, new_K)

    # Step 2: undistort new camera
    pn = _undistort(pn, new_d, new_valid)

    # Step 3: rotate from new camera to old camera
    R_rel = old_R @ new_R.T
    ones = torch.ones_like(pn[..., :1])
    pn_3d = torch.cat([pn, ones], dim=-1)
    rotated = torch.einsum('ij,...j->...i', R_rel, pn_3d)
    pn = coordframes.project(rotated)

    # Step 4: distort old camera
    pn = _distort(pn, old_d, old_valid)

    # Step 5: apply old camera intrinsics
    p_old = coordframes.apply_intrinsics(pn, old_K)
    return p_old


def _undistort(pn, d, valid_info):
    """Apply undistortion. Receives pre-cached validity info."""
    if d is None:
        return pn
    if d.shape[-1] == 4:
        ru, rd = valid_info
        return dist_module.undistort_fisheye(pn, d, ru, rd)
    else:
        return dist_module.undistort_brown_conrady(pn, d, valid_info)


def _distort(pn, d, valid_info):
    """Apply forward distortion. Receives pre-cached validity info."""
    if d is None:
        return pn
    if d.shape[-1] == 4:
        ru, _ = valid_info
        return dist_module.distort_fisheye(pn, d, ru_valid=ru)
    else:
        return dist_module.distort_brown_conrady(pn, d, valid_region=valid_info)


# =============================================================================
# Precomputation: Brown-Conrady undistortion (2D grid + Jacobian correction)
# =============================================================================

def _precompute_undistort_bc(d, fov_degrees_max, res):
    """Precompute BC undistortion lookup table (res x res x 6).

    Stores undistorted coords + undistortion Jacobian on a regular grid.
    Like CPU numba precomp_maps_undistort.
    """
    device = d.device
    dtype = d.dtype
    d_padded = dist_module._pad_distortion_coeffs(d, 14)
    valid_region = validity.brown_conrady_valid_region(d_padded)
    _, (rd, _) = valid_region

    c = (res - 1) * 0.5
    f1 = c / rd.max()
    half_fov = torch.deg2rad(torch.tensor(fov_degrees_max * 0.5, device=device, dtype=dtype))
    f2 = c / torch.tan(half_fov)
    f = torch.maximum(f1, f2)

    K_lut = torch.zeros(3, 3, device=device, dtype=dtype)
    K_lut[0, 0] = K_lut[1, 1] = f
    K_lut[0, 2] = K_lut[1, 2] = c
    K_lut[2, 2] = 1

    grid = coordframes.make_pixel_grid(res, res, device=device, dtype=dtype)
    pn = coordframes.undo_intrinsics(grid, K_lut).reshape(-1, 2)

    # Undistort
    pun = dist_module.undistort_brown_conrady(pn, d_padded, valid_region)

    # Forward distortion Jacobian at undistorted points
    _, jac_fwd = dist_module.distort_brown_conrady_with_jacobian(pun, d_padded)
    # jac_fwd: (N, 2, 2), d(pd)/d(pu). Invert to get d(pu)/d(pd).
    j00 = jac_fwd[..., 0, 0]
    j01 = jac_fwd[..., 0, 1]
    j10 = jac_fwd[..., 1, 0]
    j11 = jac_fwd[..., 1, 1]
    det = j00 * j11 - j01 * j10
    inv_det = 1.0 / det
    jac_inv = torch.stack([
        j11 * inv_det, -j01 * inv_det,
        -j10 * inv_det, j00 * inv_det,
    ], dim=-1)  # (N, 4)

    maps = torch.cat([pun, jac_inv], dim=-1).reshape(res, res, 6)
    return dict(kind='bc_undistort', maps=maps, f=f)


# =============================================================================
# Precomputation: fisheye undistortion (1D scale factor table)
# =============================================================================

def _precompute_undistort_fisheye(d, res):
    """Precompute fisheye undistortion: 1D table mapping r_d² → scale factor.

    Like CPU numba precomp_map_undistort_fisheye.
    """
    device = d.device
    dtype = d.dtype
    ru_valid, rd_valid = validity.fisheye_valid_r_max(d)
    d0, d1, d2, d3 = d[0], d[1], d[2], d[3]

    # 1D grid of distorted radii squared
    td2 = torch.linspace(
        0, (rd_valid * rd_valid).item(), res, device=device, dtype=dtype)
    td = torch.sqrt(td2)

    # Initial guess, clamped
    max_initial_t = torch.atan(ru_valid) * 0.95
    t = torch.minimum(td, max_initial_t)

    # Newton iterations to solve: t*(1 + t²*(d0 + ...)) = td
    for _ in range(4):
        t2 = t * t
        f_val = t * (1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3)))) - td
        fp = 1 + t2 * (3 * d0 + t2 * (5 * d1 + t2 * (7 * d2 + 9 * t2 * d3)))
        t = t - f_val / fp

    # Scale factor: undistorted_r / distorted_r = tan(t) / td
    scale = torch.tan(t) / td
    scale[0] = 1.0

    return dict(
        kind='fisheye_undistort', scale_map=scale,
        r_max_sq=rd_valid * rd_valid, rd_valid=rd_valid)


# =============================================================================
# Applying precomputed lookup tables
# =============================================================================

def _apply_undistort_precomp(pn, precomp):
    """Apply precomputed undistortion lookup."""
    if precomp is None:
        return pn
    if precomp['kind'] == 'bc_undistort':
        return _apply_bc_map(pn, precomp['maps'], precomp['f'])
    else:
        return _apply_fisheye_map(pn, precomp)



def _apply_bc_map(pn, maps, f):
    """Apply BC undistortion lookup with nearest-neighbor + Jacobian correction.

    Like CPU numba apply_distortion_map_inplace.

    Args:
        pn: (..., 2) distorted normalized coordinates
        maps: (res, res, 6) [pu_x, pu_y, j00, j01, j10, j11]
        f: scale factor

    Returns:
        (..., 2) undistorted normalized coordinates
    """
    res = maps.shape[0]
    c = (res - 1) * 0.5
    c_p_half = res * 0.5

    orig_shape = pn.shape
    pn_flat = pn.reshape(-1, 2)

    # Map normalized coords to lookup table pixel indices
    map_x = torch.floor(pn_flat[:, 0] * f + c_p_half)
    map_y = torch.floor(pn_flat[:, 1] * f + c_p_half)
    imap_x = map_x.long()
    imap_y = map_y.long()

    in_bounds = (map_x >= 0) & (map_y >= 0) & (imap_x < res) & (imap_y < res)
    ix = imap_x.clamp(0, res - 1)
    iy = imap_y.clamp(0, res - 1)

    looked_up = maps[iy, ix]  # (N, 6)
    pux = looked_up[:, 0]
    puy = looked_up[:, 1]
    j00 = looked_up[:, 2]
    j01 = looked_up[:, 3]
    j10 = looked_up[:, 4]
    j11 = looked_up[:, 5]

    # Sub-cell offset in normalized coordinates
    inv_f = 1.0 / f
    dx = pn_flat[:, 0] - (map_x - c) * inv_f
    dy = pn_flat[:, 1] - (map_y - c) * inv_f

    # Linear correction using undistortion Jacobian
    result_x = pux + dx * j00 + dy * j01
    result_y = puy + dx * j10 + dy * j11

    result = torch.stack([result_x, result_y], dim=-1)
    result = torch.where(in_bounds.unsqueeze(-1), result, float('nan'))
    return result.reshape(orig_shape)


def _apply_fisheye_map(pn, precomp):
    """Apply fisheye 1D radial scale factor lookup.

    Like CPU numba apply_fisheye_map_inplace.

    Args:
        pn: (..., 2) normalized coordinates
        precomp: dict with 'scale_map' (res,), 'r_max_sq'

    Returns:
        (..., 2) scaled coordinates
    """
    scale_map = precomp['scale_map']
    r_max_sq = precomp['r_max_sq']
    res = scale_map.shape[0]

    r2 = (pn * pn).sum(dim=-1)

    # Nearest-neighbor lookup into 1D table (matches CPU numba)
    idx = (r2 / r_max_sq * (res - 1)).round().long().clamp(0, res - 1)
    scale = scale_map[idx]

    result = pn * scale.unsqueeze(-1)
    result = torch.where((r2 >= r_max_sq).unsqueeze(-1), float('nan'), result)
    return result


# =============================================================================
# Batched intrinsics helpers
# =============================================================================

def _batched_undo_intrinsics(grid, Ks):
    """Undo intrinsics with per-element K: grid (h, w, 2), Ks (B, 3, 3) → (B, h, w, 2)."""
    B = Ks.shape[0]
    fx = Ks[:, 0, 0].reshape(B, 1, 1)
    fy = Ks[:, 1, 1].reshape(B, 1, 1)
    cx = Ks[:, 0, 2].reshape(B, 1, 1)
    cy = Ks[:, 1, 2].reshape(B, 1, 1)
    skew = Ks[:, 0, 1].reshape(B, 1, 1)
    y_n = (grid[..., 1] - cy) / fy
    x_n = (grid[..., 0] - cx - skew * y_n) / fx
    return torch.stack([x_n, y_n], dim=-1)


def _batched_apply_intrinsics(pn, Ks):
    """Apply intrinsics with per-element K: pn (B, h, w, 2), Ks (B, 3, 3) → (B, h, w, 2)."""
    B = Ks.shape[0]
    fx = Ks[:, 0, 0].reshape(B, 1, 1)
    fy = Ks[:, 1, 1].reshape(B, 1, 1)
    cx = Ks[:, 0, 2].reshape(B, 1, 1)
    cy = Ks[:, 1, 2].reshape(B, 1, 1)
    skew = Ks[:, 0, 1].reshape(B, 1, 1)
    px = fx * pn[..., 0] + skew * pn[..., 1] + cx
    py = fy * pn[..., 1] + cy
    return torch.stack([px, py], dim=-1)


# =============================================================================
# Grid normalization
# =============================================================================

def _normalize_grid(grid_px, old_h, old_w):
    """Convert pixel coordinates to grid_sample [-1, 1] normalized format.

    Uses align_corners=True convention: -1 maps to pixel center 0, +1 maps to
    pixel center (size-1).
    """
    x = 2.0 * grid_px[..., 0] / (old_w - 1) - 1.0
    y = 2.0 * grid_px[..., 1] / (old_h - 1) - 1.0
    return torch.stack([x, y], dim=-1)
