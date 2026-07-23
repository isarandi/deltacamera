import torch

from . import distortion as dist_module

_PI = 3.141592653589793
_TWO_PI = 2.0 * _PI


# =============================================================================
# Public API
# =============================================================================

def brown_conrady_valid_region(
    d, limit=5.0, n_vertices=128, n_vertices_coarse=24,
    n_steps_line_search=100, n_iter_newton=3,
):
    """Find the valid distortion region boundary in polar coordinates.

    Args:
        d: (C,) distortion coefficients, C <= 14
        limit: max radius to search
        n_vertices: dense angular resolution
        n_vertices_coarse: coarse angular resolution for line search
        n_steps_line_search: radial steps per angle
        n_iter_newton: Newton refinement iterations

    Returns:
        ((r_undist, theta_undist), (r_dist, theta_dist)) — polar boundary with
        wrap-around padding for correct interpolation.
    """
    d = dist_module._pad_distortion_coeffs(d, 14)
    device = d.device
    dtype = d.dtype

    # Limit search to rational model asymptote (denominator zero). Back off
    # slightly from the exact root: evaluating at the pole itself makes the Jacobian
    # determinant blow up and corrupts the boundary search.
    asymptote_limit = torch.sqrt(
        solve_cubic_smallest_nonneg_root(d[7], d[6], d[5], 1.0)) * 0.999
    limit = torch.minimum(
        torch.tensor(limit, device=device, dtype=dtype), asymptote_limit)

    # Coarse line search
    theta_coarse = torch.linspace(-_PI, _PI, n_vertices_coarse + 1,
                                  device=device, dtype=dtype)[:-1]
    radii_coarse = _line_search(theta_coarse, d, n_steps_line_search, limit)

    # Wrap for interpolation
    theta_coarse_wrap = torch.cat([theta_coarse, theta_coarse[:1] + _TWO_PI])
    radii_coarse_wrap = torch.cat([radii_coarse, radii_coarse[:1]])

    # Interpolate to dense sampling
    theta_dense = torch.linspace(-_PI, _PI, n_vertices + 1,
                                 device=device, dtype=dtype)[:-1]
    radii_dense = dist_module.interp_1d(theta_dense, theta_coarse_wrap, radii_coarse_wrap)

    # Replace non-finite with limit, refine all with Newton, restore those positions.
    # Non-finite includes NaN: interp_1d's lerp turns inf sample values (angles where the
    # line search found no fold) into NaN via inf - inf, unlike np.interp which keeps inf.
    is_inf = ~torch.isfinite(radii_dense)
    radii_init = torch.where(is_inf, limit, radii_dense)
    radii_refined = _newton_refine(radii_init, theta_dense, d, n_iter_newton)
    radii_dense = torch.where(is_inf, limit, radii_refined)
    radii_dense = radii_dense.clamp(min=torch.tensor(0, device=device, dtype=dtype), max=limit)

    # The sensor tilt (tau_x, tau_y) is a projective map applied after the 12-coefficient
    # distortion. Its Jacobian blows up (rather than crossing zero) at the tilt horizon, so
    # the det-based search above runs past it. Shrink each boundary radius until the
    # 12-coefficient distorted point stays clearly on the visible side of the horizon,
    # otherwise the boundary polygon folds over when mapped through the tilt.
    if d.shape[0] > 12 and (d[12] != 0 or d[13] != 0):
        t20 = torch.sin(d[13])
        t21 = -torch.cos(d[13]) * torch.sin(d[12])
        t22 = torch.cos(d[12]) * torch.cos(d[13])
        w_min = 0.1 * t22
        d12 = d[:12]
        for _ in range(100):
            pu12 = _polar_to_cartesian(radii_dense, theta_dense)
            q = dist_module.distort_brown_conrady(pu12, d12, valid_region=None)
            w = t20 * q[..., 0] + t21 * q[..., 1] + t22
            needs_shrink = w < w_min
            if not needs_shrink.any():
                break
            radii_dense = torch.where(needs_shrink, radii_dense * 0.95, radii_dense)

    # Map boundary to distorted space
    pu = _polar_to_cartesian(radii_dense, theta_dense)
    pd = dist_module.distort_brown_conrady(pu, d, valid_region=None)
    radii_dist, theta_dist = _cartesian_to_polar(pd)

    # Sort by distorted angle
    order = torch.argsort(theta_dist)
    radii_dist = radii_dist[order]
    theta_dist = theta_dist[order]

    # Add wrap-around padding
    theta_dense = torch.cat([theta_dense[-1:] - _TWO_PI, theta_dense, theta_dense[:1] + _TWO_PI])
    radii_dense = torch.cat([radii_dense[-1:], radii_dense, radii_dense[:1]])
    theta_dist = torch.cat([theta_dist[-1:] - _TWO_PI, theta_dist, theta_dist[:1] + _TWO_PI])
    radii_dist = torch.cat([radii_dist[-1:], radii_dist, radii_dist[:1]])

    return (radii_dense, theta_dense), (radii_dist, theta_dist)


def fisheye_valid_r_max(d):
    """Find the maximum valid radius for fisheye distortion.

    Args:
        d: (4,) fisheye coefficients

    Returns:
        (ru_valid, rd_valid) — max undistorted and distorted radii.
        ru_valid is inf if distortion is monotonic everywhere.
    """
    d0, d1, d2, d3 = d[0], d[1], d[2], d[3]
    device = d.device
    dtype = d.dtype

    n_steps = 32
    half_pi = _PI / 2
    t_start = 0.1

    # Vectorized line search: evaluate derivative at all sample points at once
    t_vals = torch.linspace(t_start, half_pi, n_steps, device=device, dtype=dtype)
    t2_vals = t_vals * t_vals
    derivs = 1 + t2_vals * (3*d0 + t2_vals * (5*d1 + t2_vals * (7*d2 + t2_vals * 9*d3)))

    # Find first negative derivative
    is_neg = derivs < 0
    has_neg = is_neg.any()

    # Index of first negative (n_steps if none found, via large sentinel)
    indices = torch.arange(n_steps, device=device)
    first_neg_idx = torch.where(is_neg, indices, n_steps).min()

    # Initial guess: the sample point just before the first negative
    prev_idx = (first_neg_idx - 1).clamp(min=0)
    t_init = torch.where(first_neg_idx > 0, t_vals[prev_idx], t_vals[0] * 0.0)

    # Newton refinement (always runs 5 iters; result unused if no root found)
    t = t_init
    for _ in range(5):
        t2 = t * t
        deriv = 1 + t2 * (3*d0 + t2 * (5*d1 + t2 * (7*d2 + t2 * 9*d3)))
        deriv2 = t * (6*d0 + t2 * (20*d1 + t2 * (42*d2 + t2 * 72*d3)))
        t = t - deriv / (deriv2 + 1e-30)

    t2 = t * t
    ru_found = torch.tan(t)
    rd_found = t * (1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3))))

    # If no negative derivative found, valid up to pi/2 → ru = inf
    t_max = torch.tensor(half_pi, device=device, dtype=dtype)
    t2_max = t_max * t_max
    rd_notfound = t_max * (1 + t2_max * (d0 + t2_max * (d1 + t2_max * (d2 + t2_max * d3))))

    ru = torch.where(has_neg, ru_found, float('inf'))
    rd = torch.where(has_neg, rd_found, rd_notfound)
    return ru, rd


def is_in_valid_region(points, r_valid, t_valid):
    """Check if points lie within a polar-defined valid region.

    Args:
        points: (..., 2)
        r_valid: (N,) valid radii at sorted angles t_valid
        t_valid: (N,) sorted angles (with wrap-around)

    Returns:
        (...) bool tensor
    """
    x, y = points[..., 0], points[..., 1]
    r = torch.hypot(x, y)
    t = torch.atan2(y, x)
    r_max = dist_module.interp_1d(t, t_valid, r_valid)
    return r <= r_max


# =============================================================================
# Jacobian determinant of distortion in polar coordinates
# =============================================================================

def jacobian_det_polar(r, t, d):
    """Jacobian determinant of Brown-Conrady distortion at polar coords (r, t).

    Zero crossings mark the boundary where distortion becomes non-invertible.

    Args:
        r: (...) radii
        t: (...) angles
        d: (14,) distortion coefficients (padded)

    Returns:
        (...) determinant values. Computed in float64 for precision.
    """
    d = dist_module._pad_distortion_coeffs(d, 14)
    tau_x, tau_y = d[12], d[13]
    sy = torch.sin(tau_y).to(torch.float64)
    cy_sx = (torch.cos(tau_y) * torch.sin(tau_x)).to(torch.float64)
    cy_cx = (torch.cos(tau_y) * torch.cos(tau_x)).to(torch.float64)
    return _jacobian_det_polar_impl(r, t, d[:12], sy, cy_sx, cy_cx)


def _jacobian_det_polar_impl(r, t, d, sy, cy_sx, cy_cx):
    """Core Jacobian determinant computation. All intermediates in float64."""
    r = r.to(torch.float64)
    t = t.to(torch.float64)

    k0 = d[0].to(torch.float64)
    k1 = d[1].to(torch.float64)
    k2 = d[2].to(torch.float64)
    k3 = d[3].to(torch.float64)
    k4 = d[4].to(torch.float64)
    k5 = d[5].to(torch.float64)
    k6 = d[6].to(torch.float64)
    k7 = d[7].to(torch.float64)
    k8 = d[8].to(torch.float64)
    k9 = d[9].to(torch.float64)
    k10 = d[10].to(torch.float64)
    k11 = d[11].to(torch.float64)

    x0 = torch.sin(t)
    x1 = torch.cos(t)
    x2 = 2.0 * r
    x3 = x2 * (k2 * x1 - k3 * x0)
    x4 = k2 * x0
    x5 = k3 * x1
    x6 = r * r
    x7 = k1 + k4 * x6
    x8 = k0 + x6 * x7
    x9 = x6 * x8 + 1.0
    x10 = k6 + k7 * x6
    x11 = k5 + x10 * x6
    x13 = 1.0 / (x11 * x6 + 1.0)
    x23 = x4 + x5
    x24 = x13 * x9
    x14 = x24 + x2 * x23
    x15 = x1 * x14
    x16 = k9 * x6
    x17 = k3 + k8
    x18 = 2.0 * r * x6
    x19 = (
        x13 * (x2 * x8 + x6 * (k4 * x18 + x2 * x7))
        + 2.0 * x23
        - x13 * x24 * (x11 * x2 + x6 * (k7 * x18 + x10 * x2))
    )
    x20 = x0 * x14
    x21 = k11 * x6
    x22 = k10 + k2

    det_12 = (
        (x0 * x3 + x15) * (r * (4.0 * x16 + 2.0 * x17 + x1 * x19) + x15)
        - (x1 * x3 - x20) * (r * (4.0 * x21 + 2.0 * x22 + x0 * x19) + x20)
    )

    # Compute distorted point for tilt
    a = (x6 * (k0 + x6 * (k1 + k4 * x6)) + 1.0) / (x6 * (k5 + x6 * (k6 + k7 * x6)) + 1.0)
    b = 2.0 * (k3 * r * x1 + k2 * r * x0)
    c1 = x6 * (k3 + k8 + k9 * x6)
    c2 = x6 * (k2 + k10 + k11 * x6)
    x_d = r * x1 * (a + b) + c1
    y_d = r * x0 * (a + b) + c2

    z = sy * x_d - cy_sx * y_d + cy_cx
    # det(J_tilt) = det(M_tilt) / z^3 = (cy*cx)^2 / z^3
    inv_z = 1.0 / z
    det_tilt = cy_cx * cy_cx * inv_z * inv_z * inv_z

    return det_12 * det_tilt


def jacobian_det_and_deriv_polar(r, t, d):
    """Jacobian determinant and its derivative w.r.t. r.

    Used for Newton optimization to find the exact boundary.

    Args:
        r: (...) radii
        t: (...) angles
        d: (14,) distortion coefficients (padded)

    Returns:
        (fval, deriv) — both (...) tensors
    """
    d = dist_module._pad_distortion_coeffs(d, 14)
    tau_x, tau_y = d[12], d[13]
    sy = torch.sin(tau_y).to(torch.float64)
    cy_sx = (torch.cos(tau_y) * torch.sin(tau_x)).to(torch.float64)
    cy_cx = (torch.cos(tau_y) * torch.cos(tau_x)).to(torch.float64)
    return _jacobian_det_and_deriv_polar_impl(r, t, d[:12], sy, cy_sx, cy_cx)


def _jacobian_det_and_deriv_polar_impl(r, t, d, sy, cy_sx, cy_cx):
    """Core Jacobian determinant + derivative computation in float64."""
    r = r.to(torch.float64)
    t = t.to(torch.float64)

    k0 = d[0].to(torch.float64)
    k1 = d[1].to(torch.float64)
    k2 = d[2].to(torch.float64)
    k3 = d[3].to(torch.float64)
    k4 = d[4].to(torch.float64)
    k5 = d[5].to(torch.float64)
    k6 = d[6].to(torch.float64)
    k7 = d[7].to(torch.float64)
    k8 = d[8].to(torch.float64)
    k9 = d[9].to(torch.float64)
    k10 = d[10].to(torch.float64)
    k11 = d[11].to(torch.float64)

    x0 = torch.sin(t)
    x1 = torch.cos(t)
    x2 = k2 * x1
    x3 = k3 * x0
    x4 = 2.0 * r
    x47 = x2 - x3
    x5 = x4 * x47
    x6 = k2 * x0
    x7 = k3 * x1
    x8 = r * r
    x9 = k4 * x8
    x10 = k1 + x9
    x11 = x10 * x8
    x12 = k0 + x11
    x13 = x12 * x8 + 1.0
    x14 = k7 * x8
    x15 = k6 + x14
    x16 = x15 * x8
    x17 = k5 + x16
    x19 = 1.0 / (x17 * x8 + 1.0)
    x48 = x6 + x7
    x20 = x13 * x19 + x4 * x48
    x21 = x1 * x20
    x22 = x0 * x5 + x21
    x23 = k9 * x8
    x24 = k3 + k8
    x25 = 2.0 * r * x8
    x26 = k4 * x25 + x10 * x4
    x27 = x26 * x8
    x28 = k7 * x25 + x15 * x4
    x29 = x28 * x8
    x30 = -2.0 * r * x17 - x29
    x31 = x19 * x19
    x32 = x13 * x31
    x49 = x32 * x30
    x33 = x19 * (x12 * x4 + x27) + x49 + 2.0 * x48
    x34 = x1 * x33
    x35 = x0 * x20
    x36 = x1 * x5 - x35
    x37 = k11 * x8
    x38 = k10 + k2
    x39 = x0 * x33
    x40 = 2.0 * x47
    x41 = 2.0 * (k10 + k2)
    x42 = 2.0 * (k3 + k8)
    x43 = 6.0 * r
    x44 = 2.0 * x8
    x45 = 4.0 * r
    x46 = (
        -x49 * (x17 * x45 + 2.0 * x29) * x19
        + x19 * (2.0 * (k0 + x11) + x26 * x45 + x44 * (k1 + 6.0 * x9))
        + x30 * x31 * (x12 * x45 + 2.0 * x27)
        - x32 * (2.0 * (k5 + x16) + x28 * x45 + x44 * (k6 + 6.0 * x14))
    )

    fval_12 = (
        x22 * (r * (4.0 * x23 + 2.0 * x24 + x34) + x21)
        - x36 * (r * (4.0 * x37 + 2.0 * x38 + x39) + x35)
    )
    deriv_12 = (
        x22 * (r * (k9 * x43 + x1 * x46) + 6.0 * x23 + 2.0 * x34 + x42)
        - x36 * (r * (k11 * x43 + x0 * x46) + 6.0 * x37 + 2.0 * x39 + x41)
        + (r * (4.0 * x23 + x34 + x42) + x21) * (x0 * x40 + x34)
        - (r * (4.0 * x37 + x39 + x41) + x35) * (x1 * x40 - x39)
    )

    # Compute distorted point and its derivative w.r.t. r
    numer = x8 * (k0 + x8 * (k1 + k4 * x8)) + 1.0
    denom = x8 * (k5 + x8 * (k6 + k7 * x8)) + 1.0
    a = numer / denom
    b = 2.0 * (k3 * r * x1 + k2 * r * x0)
    c1 = x8 * (k3 + k8 + k9 * x8)
    c2 = x8 * (k2 + k10 + k11 * x8)
    s = a + b
    x_d = r * x1 * s + c1
    y_d = r * x0 * s + c2

    k2_p_k10 = k2 + k10
    k3_p_k8 = k3 + k8

    r4 = x8 * x8
    dnumer_dr2 = k0 + 2.0 * k1 * x8 + 3.0 * k4 * r4
    ddenom_dr2 = k5 + 2.0 * k6 * x8 + 3.0 * k7 * r4
    da_dr = (dnumer_dr2 * denom - numer * ddenom_dr2) / (denom * denom) * 2.0 * r

    db_dr = 2.0 * (k3 * x1 + k2 * x0)
    dc1_dr = 2.0 * r * (k3_p_k8 + 2.0 * k9 * x8)
    dc2_dr = 2.0 * r * (k2_p_k10 + 2.0 * k11 * x8)

    ds_dr = da_dr + db_dr
    dx_d_dr = x1 * s + r * x1 * ds_dr + dc1_dr
    dy_d_dr = x0 * s + r * x0 * ds_dr + dc2_dr

    z = sy * x_d - cy_sx * y_d + cy_cx
    dz_dr = sy * dx_d_dr - cy_sx * dy_d_dr

    # det(J_tilt) = det(M_tilt) / z^3 = (cy*cx)^2 / z^3
    det_M = cy_cx * cy_cx
    inv_z = 1.0 / z
    inv_z3 = inv_z * inv_z * inv_z
    det_tilt = det_M * inv_z3
    ddet_tilt_dr = -3.0 * det_M * inv_z3 * inv_z * dz_dr

    fval = fval_12 * det_tilt
    deriv = deriv_12 * det_tilt + fval_12 * ddet_tilt_dr

    return fval, deriv


# =============================================================================
# Cubic solver (Cardano's formula)
# =============================================================================

def solve_cubic_smallest_nonneg_root(a, b, c, d_coeff):
    """Solve ax^3 + bx^2 + cx + d = 0, return smallest non-negative root or inf.

    Mirrors the numpy implementation in validity.py: computed in float64, a leading
    coefficient that is tiny relative to the others (common case: the k6 rational
    distortion coefficient) is handled by a quadratic branch, near-double roots take
    the three-real-roots branch via a discriminant tolerance, and all candidate roots
    are Newton-polished on the full cubic before selection. Returns a float64 scalar
    tensor on the same device.
    """
    device = _get_device(a, b, c, d_coeff)
    a = torch.as_tensor(a, dtype=torch.float64, device=device)
    b = torch.as_tensor(b, dtype=torch.float64, device=device)
    c = torch.as_tensor(c, dtype=torch.float64, device=device)
    d = torch.as_tensor(d_coeff, dtype=torch.float64, device=device)
    inf = torch.tensor(float('inf'), dtype=torch.float64, device=device)

    candidates = []
    if abs(a) <= 1e-3 * max(abs(b), abs(c), abs(d)):
        # Small leading coefficient: Cardano's h discriminant cancels catastrophically
        # (even in float64 once |a| is ~1e-8 of the rest), so additionally seed the
        # near-quadratic root estimates and the far root ~ -b/a. All candidates are
        # Newton-polished and residual-checked below, so inexact or duplicate seeds
        # (in the overlap region where Cardano also works) are harmless.
        if b == 0:
            if c != 0:
                candidates.append(-d / c)
        else:
            D = c * c - 4 * b * d
            if D >= 0:
                sqrt_D = torch.sqrt(D)
                candidates.append((-c + sqrt_D) / (2 * b))
                candidates.append((-c - sqrt_D) / (2 * b))
            if a != 0:
                candidates.append(-b / a)

    if a != 0:
        f = (3 * c / a - (b * b) / (a * a)) / 3
        g = (2 * (b * b * b) / (a * a * a) - 9 * b * c / (a * a) + 27 * d / a) / 27
        h = g * g / 4 + f * f * f / 27
        # Relative tolerance on the discriminant: near-double roots must take the
        # three-real-roots branch, else a positive root can be missed entirely.
        h_scale = g * g / 4 + abs(f * f * f) / 27

        if f == 0 and g == 0 and h == 0:
            candidates.append(-b / (3 * a))
        elif h <= 1e-12 * h_scale:
            i_val = torch.sqrt(torch.clamp(g * g / 4 - h, min=0.0))
            if i_val != 0:
                k = torch.acos(torch.clamp(-g / (2 * i_val), -1.0, 1.0))
            else:
                k = torch.zeros_like(g)
            j = i_val.pow(1.0 / 3.0)
            M = torch.cos(k / 3)
            N = 1.7320508075688772 * torch.sin(k / 3)  # sqrt(3)
            P = -b / (3 * a)
            candidates.append(P + 2 * j * M)
            candidates.append(P - j * (M + N))
            candidates.append(P - j * (M - N))
        else:
            sq_h = torch.sqrt(h)
            val_s = -g / 2 + sq_h
            val_u = -g / 2 - sq_h
            S = torch.sign(val_s) * torch.abs(val_s).pow(1.0 / 3.0)
            U = torch.sign(val_u) * torch.abs(val_u).pow(1.0 / 3.0)
            candidates.append((S + U) - b / (3 * a))

    best = inf
    for x in candidates:
        if not torch.isfinite(x):
            continue
        for _ in range(8):
            fx = ((a * x + b) * x + c) * x + d
            fpx = (3 * a * x + 2 * b) * x + c
            if fpx == 0:
                break
            x = x - fx / fpx
        # Accept only converged roots: an inexact candidate (e.g. -b/a for the far
        # root) that Newton failed to converge must not masquerade as a root.
        fx = ((a * x + b) * x + c) * x + d
        fx_scale = abs(a * x * x * x) + abs(b * x * x) + abs(c * x) + abs(d)
        if abs(fx) > 1e-6 * fx_scale:
            continue
        if x >= 0 and x < best:
            best = x
    return best


def _get_device(*args):
    for a in args:
        if isinstance(a, torch.Tensor):
            return a.device
    return torch.device('cpu')


# =============================================================================
# Internal helpers
# =============================================================================

def _line_search(t, d, n_steps, limit):
    """For each angle t, find first radius where jacobian_det_polar < 0.

    Args:
        t: (N,) angles
        d: (14,) distortion coefficients
        n_steps: number of radial samples

    Returns:
        (N,) radii at which boundary lies (inf if never negative)
    """
    device = t.device
    dtype = t.dtype

    # Denser sampling near origin
    u = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)
    r_samples = (u ** 1.5) * limit  # (S,)

    # Evaluate: (N, S)
    jac_det = jacobian_det_polar(
        r_samples.unsqueeze(0), t.unsqueeze(1), d)

    # Find first negative per row
    is_negative = jac_det < 0  # (N, S)
    # For rows with no negative, return inf
    has_negative = is_negative.any(dim=1)
    # argmax on int gives first True
    first_neg_idx = torch.argmax(is_negative.int(), dim=1)  # (N,)

    # Radius just before first negative
    prev_idx = (first_neg_idx - 1).clamp(min=0)
    result = r_samples[prev_idx]

    # Where first negative is at index 0, return 0
    result = torch.where(first_neg_idx == 0, torch.zeros_like(result), result)
    # Where no negative found, return inf
    result = torch.where(has_negative, result, float('inf'))

    return result


def _newton_refine(r, t, d, n_iter):
    """Newton refinement of boundary radius.

    Args:
        r: (N,) initial radii
        t: (N,) angles
        d: (14,) coefficients
        n_iter: iterations

    Returns:
        (N,) refined radii
    """
    r_new = r.to(torch.float64)
    t_f64 = t.to(torch.float64)
    for _ in range(n_iter):
        f, df = jacobian_det_and_deriv_polar(r_new, t_f64, d)
        r_new = r_new - f / df
    return r_new.to(r.dtype)


def _polar_to_cartesian(r, t):
    """Convert polar (r, theta) to Cartesian (x, y).

    Args:
        r: (N,)
        t: (N,)

    Returns:
        (N, 2) as [x, y]
    """
    return torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=-1)


def _cartesian_to_polar(xy):
    """Convert Cartesian (x, y) to polar (r, theta).

    Args:
        xy: (N, 2)

    Returns:
        (r: (N,), t: (N,))
    """
    r = torch.hypot(xy[..., 0], xy[..., 1])
    t = torch.atan2(xy[..., 1], xy[..., 0])
    return r, t
