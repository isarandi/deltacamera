import torch


# =============================================================================
# Utilities
# =============================================================================

def interp_1d(x, xp, fp):
    """1D linear interpolation, like np.interp.

    Args:
        x: (...) query points
        xp: (N,) sorted sample x-coordinates
        fp: (N,) sample values

    Returns:
        (...) interpolated values, clamped to fp[0]/fp[-1] outside range.
    """
    idx = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    f0 = fp[idx - 1]
    f1 = fp[idx]
    t = (x - x0) / (x1 - x0)
    return f0 + t * (f1 - f0)


def _pad_distortion_coeffs(d, n=14):
    """Pad distortion coefficients to length n with zeros."""
    if d.shape[-1] >= n:
        return d[..., :n]
    pad_size = n - d.shape[-1]
    zeros = torch.zeros(*d.shape[:-1], pad_size, dtype=d.dtype, device=d.device)
    return torch.cat([d, zeros], dim=-1)


def _hypot_safe(x, y):
    """torch.hypot with a well-defined zero gradient at the exact origin.

    The partials of hypot at (0, 0) are 0/0 = NaN, and a NaN partial poisons the
    backward pass even through an unselected torch.where branch, so the input is
    sanitized before the op (divide-no-nan pattern).
    """
    at_origin = (x == 0) & (y == 0)
    x_safe = torch.where(at_origin, torch.ones_like(x), x)
    return torch.where(at_origin, torch.zeros_like(x), torch.hypot(x_safe, y))


def _atan2_safe(y, x):
    """torch.atan2 with a well-defined zero gradient at the exact origin (angle 0 there).

    Same rationale as _hypot_safe: the partials of atan2 at (0, 0) are NaN.
    """
    at_origin = (x == 0) & (y == 0)
    x_safe = torch.where(at_origin, torch.ones_like(x), x)
    return torch.where(at_origin, torch.zeros_like(y), torch.atan2(y, x_safe))


# =============================================================================
# Tilt matrix helpers (14-param Brown-Conrady)
# =============================================================================

def _compute_tilt_matrix(tau_x, tau_y):
    """Compute 3x3 tilt matrix from tilt angles.

    Returns:
        (3, 3) tilt matrix
    """
    cx = torch.cos(tau_x)
    sx = torch.sin(tau_x)
    cy = torch.cos(tau_y)
    sy = torch.sin(tau_y)
    zero = torch.zeros_like(cx)
    row0 = torch.stack([cx, zero, zero])
    row1 = torch.stack([-sx * sy, cy, zero])
    row2 = torch.stack([sy, -cy * sx, cy * cx])
    return torch.stack([row0, row1, row2])


def _compute_inv_tilt_matrix(tau_x, tau_y):
    """Compute inverse tilt matrix from tilt angles.

    Returns:
        (3, 3) inverse tilt matrix
    """
    cx = torch.cos(tau_x)
    sx = torch.sin(tau_x)
    cy = torch.cos(tau_y)
    sy = torch.sin(tau_y)
    inv_cx = 1.0 / cx
    inv_cy = 1.0 / cy
    inv_cxcy = inv_cx * inv_cy
    zero = torch.zeros_like(cx)
    row0 = torch.stack([inv_cx, zero, zero])
    row1 = torch.stack([sy * sx * inv_cxcy, inv_cy, zero])
    row2 = torch.stack([-sy * inv_cy, sx * inv_cxcy, inv_cxcy])
    return torch.stack([row0, row1, row2])


def _apply_tilt(points, tau_x, tau_y):
    """Apply tilt transformation with perspective division.

    Args:
        points: (..., 2) distorted points (before tilt)
        tau_x, tau_y: tilt angles (scalar tensors)

    Returns:
        (..., 2) tilted points
    """
    T = _compute_tilt_matrix(tau_x, tau_y)
    x, y = points[..., 0], points[..., 1]
    w = T[2, 0] * x + T[2, 1] * y + T[2, 2]
    x_out = (T[0, 0] * x + T[0, 1] * y + T[0, 2]) / w
    y_out = (T[1, 0] * x + T[1, 1] * y + T[1, 2]) / w
    return torch.stack([x_out, y_out], dim=-1)


def _undo_tilt(points, tau_x, tau_y):
    """Remove tilt transformation with perspective division.

    Args:
        points: (..., 2) tilted distorted points
        tau_x, tau_y: tilt angles (scalar tensors)

    Returns:
        (..., 2) untilted points
    """
    T_inv = _compute_inv_tilt_matrix(tau_x, tau_y)
    x, y = points[..., 0], points[..., 1]
    w = T_inv[2, 0] * x + T_inv[2, 1] * y + T_inv[2, 2]
    x_out = (T_inv[0, 0] * x + T_inv[0, 1] * y + T_inv[0, 2]) / w
    y_out = (T_inv[1, 0] * x + T_inv[1, 1] * y + T_inv[1, 2]) / w
    return torch.stack([x_out, y_out], dim=-1)


# =============================================================================
# Polar region helpers (for validity checking/clipping)
# =============================================================================

def _clip_to_polar_region(points, r_valid, t_valid):
    """Scale points radially inward to stay within a polar-defined valid region.

    Args:
        points: (..., 2)
        r_valid: (N,) valid radii at angles t_valid
        t_valid: (N,) sorted angles

    Returns:
        (..., 2) clipped points
    """
    x, y = points[..., 0], points[..., 1]
    r = _hypot_safe(x, y)
    t = _atan2_safe(y, x)
    r_max = interp_1d(t, t_valid, r_valid)
    scale = torch.where(r > r_max, r_max / r.clamp(min=1e-12), torch.ones_like(r))
    return points * scale.unsqueeze(-1)


def _invalidate_outside_polar_region(points, r_valid, t_valid):
    """Set points outside valid region to NaN.

    Args:
        points: (..., 2)
        r_valid: (N,) valid radii at angles t_valid
        t_valid: (N,) sorted angles

    Returns:
        (..., 2) with NaN for invalid points
    """
    x, y = points[..., 0], points[..., 1]
    r = _hypot_safe(x, y)
    t = _atan2_safe(y, x)
    r_max = interp_1d(t, t_valid, r_valid)
    invalid = r > r_max
    return torch.where(invalid.unsqueeze(-1), float('nan'), points)


# =============================================================================
# Brown-Conrady distortion (12/14-param)
# =============================================================================

def distort_brown_conrady(pu, d, valid_region=None):
    """Apply Brown-Conrady forward distortion.

    Args:
        pu: (..., 2) undistorted normalized points
        d: (C,) distortion coefficients, C <= 14. Padded to 14 internally.
        valid_region: optional ((r_u, t_u), (r_d, t_d)) polar boundary

    Returns:
        (..., 2) distorted points. NaN for points outside valid region.
    """
    d = _pad_distortion_coeffs(d, 14)
    k0, k1, k2, k3, k4 = d[0], d[1], d[2], d[3], d[4]
    k5, k6, k7, k8, k9 = d[5], d[6], d[7], d[8], d[9]
    k10, k11 = d[10], d[11]
    tau_x, tau_y = d[12], d[13]

    x = pu[..., 0]
    y = pu[..., 1]
    r2 = x * x + y * y

    numer = r2 * (k0 + r2 * (k1 + k4 * r2)) + 1
    denom = r2 * (k5 + r2 * (k6 + k7 * r2)) + 1
    a = numer / denom
    b = 2 * k2 * y + 2 * k3 * x
    cx = r2 * (k3 + k8 + k9 * r2)
    cy = r2 * (k2 + k10 + k11 * r2)
    s = a + b
    xd = x * s + cx
    yd = y * s + cy
    result = torch.stack([xd, yd], dim=-1)
    result = _apply_tilt(result, tau_x, tau_y)

    if valid_region is not None:
        (ru_valid, tu_valid), _ = valid_region
        result = _invalidate_outside_polar_region_input(
            pu, result, ru_valid, tu_valid)

    return result


def _invalidate_outside_polar_region_input(input_pts, output_pts, r_valid, t_valid):
    """Set output to NaN where input points are outside valid region."""
    x, y = input_pts[..., 0], input_pts[..., 1]
    r = _hypot_safe(x, y)
    t = _atan2_safe(y, x)
    r_max = interp_1d(t, t_valid, r_valid)
    invalid = r > r_max
    return torch.where(invalid.unsqueeze(-1), float('nan'), output_pts)


def distort_brown_conrady_with_jacobian(pu, d):
    """Apply Brown-Conrady forward distortion and compute 2x2 Jacobian.

    No validity checking — caller must ensure points are valid.

    Args:
        pu: (..., 2) undistorted normalized points
        d: (C,) distortion coefficients, C <= 14

    Returns:
        (distorted: (..., 2), jacobian: (..., 2, 2))
        jacobian[..., i, j] = d(distorted_i) / d(undistorted_j)
    """
    d = _pad_distortion_coeffs(d, 14)
    k0, k1, k2, k3, k4 = d[0], d[1], d[2], d[3], d[4]
    k5, k6, k7, k8, k9 = d[5], d[6], d[7], d[8], d[9]
    k10, k11 = d[10], d[11]
    tau_x, tau_y = d[12], d[13]

    x = pu[..., 0]
    y = pu[..., 1]
    r2 = x * x + y * y
    _2x = 2 * x
    _2y = 2 * y

    # Intermediates matching CPU variable names
    k9_r2 = k9 * r2
    k11_r2 = k11 * r2
    k4_r2 = k4 * r2
    k7_r2 = k7 * r2
    x2 = k3 + k8 + k9_r2
    x16 = k2 + k10 + k11_r2
    x6 = k1 + k4_r2
    x7 = k0 + r2 * x6
    x10 = k6 + k7_r2
    x11 = k5 + r2 * x10
    x13 = 1.0 / (r2 * x11 + 1)
    x29 = x13 * (r2 * x7 + 1)
    x14 = x * 2 * k3 + y * 2 * k2 + x29
    x26 = x13 * ((r2 * (k4_r2 + x6) + x7) - x29 * (r2 * (k7_r2 + x10) + x11))
    x19 = x * x26 + k3
    x21 = y * x26 + k2
    x27 = k9_r2 + x2
    x28 = k11_r2 + x16

    # Distorted points
    xd = x * x14 + r2 * x2
    yd = y * x14 + r2 * x16

    # Jacobian of 12-param
    j00_12 = _2x * (x19 + x27) + x14
    j11_12 = _2y * (x21 + x28) + x14
    j01_12 = _2x * x21 + _2y * x27
    j10_12 = _2y * x19 + _2x * x28

    T = _compute_tilt_matrix(tau_x, tau_y)
    w = T[2, 0] * xd + T[2, 1] * yd + T[2, 2]
    inv_w = 1.0 / w
    x_out = (T[0, 0] * xd + T[0, 1] * yd + T[0, 2]) * inv_w
    y_out = (T[1, 0] * xd + T[1, 1] * yd + T[1, 2]) * inv_w

    # Jacobian of tilt w.r.t. (xd, yd)
    jt00 = (T[0, 0] - x_out * T[2, 0]) * inv_w
    jt01 = (T[0, 1] - x_out * T[2, 1]) * inv_w
    jt10 = (T[1, 0] - y_out * T[2, 0]) * inv_w
    jt11 = (T[1, 1] - y_out * T[2, 1]) * inv_w

    # Chain rule: J_total = J_tilt @ J_12param
    j00 = jt00 * j00_12 + jt01 * j10_12
    j01 = jt00 * j01_12 + jt01 * j11_12
    j10 = jt10 * j00_12 + jt11 * j10_12
    j11 = jt10 * j01_12 + jt11 * j11_12

    distorted = torch.stack([x_out, y_out], dim=-1)

    jacobian = torch.stack([
        torch.stack([j00, j01], dim=-1),
        torch.stack([j10, j11], dim=-1),
    ], dim=-2)

    return distorted, jacobian


def undistort_brown_conrady(
    pd, d, valid_region, n_iter_fp=5, n_iter_newton=2, lambda_=0.5,
):
    """Remove Brown-Conrady distortion via fixed-point + Newton iteration.

    Args:
        pd: (..., 2) distorted normalized points
        d: (C,) distortion coefficients, C <= 14
        valid_region: ((r_u, t_u), (r_d, t_d)) polar boundary arrays
        n_iter_fp: number of fixed-point iterations
        n_iter_newton: number of Newton iterations
        lambda_: damping factor for near-singular Jacobians

    Returns:
        (..., 2) undistorted points. NaN for invalid points.
    """
    d = _pad_distortion_coeffs(d, 14)
    k0, k1, k2, k3, k4 = d[0], d[1], d[2], d[3], d[4]
    k5, k6, k7, k8, k9 = d[5], d[6], d[7], d[8], d[9]
    k10, k11 = d[10], d[11]
    tau_x, tau_y = d[12], d[13]

    (ru_valid, tu_valid), (rd_valid, td_valid) = valid_region

    # Step 1: undo tilt (identity when tau=0)
    pn = _undo_tilt(pd, tau_x, tau_y)

    # Step 2: mark/filter invalid distorted points
    pn_valid = _invalidate_outside_polar_region(pn, rd_valid, td_valid)

    # Step 3: initialize estimate and clip to valid undistorted region
    pu = _clip_to_polar_region(pn_valid.clone(), ru_valid, tu_valid)

    k3_p_k8 = k3 + k8
    k2_p_k10 = k2 + k10

    # Step 4: fixed-point iterations
    for _ in range(n_iter_fp):
        pu_x = pu[..., 0]
        pu_y = pu[..., 1]
        r2 = pu_x * pu_x + pu_y * pu_y
        inv_a = (r2 * (k5 + r2 * (k6 + k7 * r2)) + 1) / (
            r2 * (k0 + r2 * (k1 + k4 * r2)) + 1)
        b = 2 * k2 * pu_y + 2 * k3 * pu_x
        cx = r2 * (k3_p_k8 + k9 * r2)
        cy = r2 * (k2_p_k10 + k11 * r2)
        new_x = (pn_valid[..., 0] - cx - pu_x * b) * inv_a
        new_y = (pn_valid[..., 1] - cy - pu_y * b) * inv_a
        pu = torch.stack([new_x, new_y], dim=-1)

    # Step 5: clip to valid region again
    pu = _clip_to_polar_region(pu, ru_valid, tu_valid)

    # Step 6: Newton iterations
    for _ in range(n_iter_newton):
        pd_hat, jac = distort_brown_conrady_with_jacobian(pu, d[:12])
        err_x = pn_valid[..., 0] - pd_hat[..., 0]
        err_y = pn_valid[..., 1] - pd_hat[..., 1]

        j00 = jac[..., 0, 0]
        j01 = jac[..., 0, 1]
        j10 = jac[..., 1, 0]
        j11 = jac[..., 1, 1]
        det = j00 * j11 - j01 * j10

        # Damp near-singular Jacobians
        damped = det.abs() < 0.05
        j00_d = torch.where(damped, j00 + lambda_, j00)
        j11_d = torch.where(damped, j11 + lambda_, j11)
        det_d = j00_d * j11_d - j01 * j10
        inv_det = 1.0 / det_d

        dx = inv_det * (j11_d * err_x - j01 * err_y)
        dy = inv_det * (j00_d * err_y - j10 * err_x)
        pu = pu + torch.stack([dx, dy], dim=-1)

    return pu


# =============================================================================
# Fisheye distortion (Kannala-Brandt, 4-param)
# =============================================================================

def distort_fisheye(pu, d, ru_valid=None):
    """Apply Kannala-Brandt fisheye forward distortion.

    Args:
        pu: (..., 2) undistorted normalized points
        d: (4,) fisheye coefficients [d0, d1, d2, d3]
        ru_valid: optional max valid undistorted radius

    Returns:
        (..., 2) distorted points
    """
    x = pu[..., 0]
    y = pu[..., 1]
    r = _hypot_safe(x, y)
    theta = torch.atan(r)
    theta2 = theta * theta
    theta_d = theta * (1 + theta2 * (d[0] + theta2 * (d[1] + theta2 * (d[2] + theta2 * d[3]))))
    # Guard the denominator: torch.where alone doesn't prevent the discarded branch's
    # division by zero from poisoning gradients (0 * inf = NaN in backward).
    r_safe = torch.where(r > 1e-12, r, torch.ones_like(r))
    scale = torch.where(r > 1e-12, theta_d / r_safe, torch.ones_like(r))
    result = pu * scale.unsqueeze(-1)

    if ru_valid is not None:
        invalid = r > ru_valid
        result = torch.where(invalid.unsqueeze(-1), float('nan'), result)

    return result


def undistort_fisheye(pd, d, ru_valid, rd_valid, n_iter=3):
    """Remove Kannala-Brandt fisheye distortion via Newton iteration.

    Args:
        pd: (..., 2) distorted normalized points
        d: (4,) fisheye coefficients [d0, d1, d2, d3]
        ru_valid: max valid undistorted radius
        rd_valid: max valid distorted radius
        n_iter: number of Newton iterations

    Returns:
        (..., 2) undistorted points
    """
    x = pd[..., 0]
    y = pd[..., 1]
    rd = _hypot_safe(x, y)

    # Mark invalid
    invalid = rd > rd_valid

    # Initial guess, clamped to safe range
    ru_val = ru_valid if isinstance(ru_valid, torch.Tensor) else torch.tensor(
        ru_valid, dtype=pd.dtype, device=pd.device)
    t = rd.clamp(max=0.95 * torch.atan(ru_val))

    # Newton iterations
    d0, d1, d2, d3 = d[0], d[1], d[2], d[3]
    for _ in range(n_iter):
        t2 = t * t
        f = t * (1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3)))) - rd
        fp = 1 + t2 * (3 * d0 + t2 * (5 * d1 + t2 * (7 * d2 + 9 * t2 * d3)))
        t = t - f / fp

    # Recover undistorted points (guarded denominator, see distort_fisheye)
    rd_safe = torch.where(rd > 1e-12, rd, torch.ones_like(rd))
    scale = torch.where(rd > 1e-12, torch.tan(t) / rd_safe, torch.ones_like(rd))
    result = pd * scale.unsqueeze(-1)

    result = torch.where(invalid.unsqueeze(-1), float('nan'), result)
    return result
