import numba
import numpy as np

import lensform.reprojection
import lensform.core
import lensform.validity


def distort_points(pu, d, check_validity=True, clip_to_valid=False, dst=None):
    if check_validity and clip_to_valid:
        raise ValueError("Only one of check_validity and clip_to_valid can be True")

    if check_validity or clip_to_valid:
        polar_ud_valid = lensform.validity.get_valid_distortion_region_cached(
            d.astype(np.float32).tobytes()
        )
    else:
        polar_ud_valid = None
    return _distort_points(
        pu,
        d,
        polar_ud_valid=polar_ud_valid,
        check_validity=check_validity,
        clip_to_valid=clip_to_valid,
        dst=dst,
    )


def undistort_points(pn, d, check_validity=True, clip_to_valid=False):
    if check_validity and clip_to_valid:
        raise ValueError("Only one of check_validity and clip_to_valid can be True")

    polar_ud = lensform.validity.get_valid_distortion_region_cached(
        d.astype(np.float32).tobytes()
    )
    return _undistort_points(
        pn,
        d,
        polar_ud,
        check_validity=check_validity,
        clip_to_valid=clip_to_valid,
        include_jacobian=False,
        n_iter_fixed_point=5,
        n_iter_newton=2,
        lambda_=5e-1,
    )


@numba.njit(error_model='numpy', cache=True)
def _distort_points(pun, d, polar_ud_valid, check_validity, clip_to_valid, dst):
    _1 = np.float32(1)
    _2 = np.float32(2)
    _nan = np.float32(np.nan)

    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = d
    k2_p_k10 = k2 + k10
    k3_p_k8 = k3 + k8
    _2_k2 = _2 * k2
    _2_k3 = _2 * k3

    if dst is None:
        dst = np.empty_like(pun)

    if check_validity and polar_ud_valid is not None:
        polar_u_valid, polar_d_valid = polar_ud_valid
        ru_valid, tu_valid = polar_u_valid
        ru2_valid_min = np.square(np.min(ru_valid))
        ru2_valid_max = np.square(np.max(ru_valid))
        for i in range(pun.shape[0]):
            pux = pun[i, 0]
            puy = pun[i, 1]
            r2 = pux * pux + puy * puy
            if r2 > ru2_valid_min:
                if r2 > ru2_valid_max:
                    dst[i, 0] = _nan
                    dst[i, 1] = _nan
                    continue

                t = np.arctan2(puy, pux)
                r_valid_interp = np.float32(np.interp(t, tu_valid, ru_valid))
                if r2 > r_valid_interp * r_valid_interp:
                    dst[i, 0] = _nan
                    dst[i, 1] = _nan
                    continue

            a = (r2 * (k0 + r2 * (k1 + k4 * r2)) + _1) / (r2 * (k5 + r2 * (k6 + k7 * r2)) + _1)
            b = _2_k2 * puy + _2_k3 * pux
            cx = r2 * (k3_p_k8 + k9 * r2)
            cy = r2 * (k2_p_k10 + k11 * r2)
            s = a + b
            dst[i, 0] = pux * s + cx
            dst[i, 1] = puy * s + cy
    elif clip_to_valid and polar_ud_valid is not None:
        polar_u_valid, polar_d_valid = polar_ud_valid
        ru_valid, tu_valid = polar_u_valid
        ru2_valid_min = np.square(np.min(ru_valid))

        for i in range(pun.shape[0]):
            pux = pun[i, 0]
            puy = pun[i, 1]
            r2 = pux * pux + puy * puy

            if r2 > ru2_valid_min:
                t = np.arctan2(puy, pux)
                r_valid_interp = np.float32(np.interp(t, tu_valid, ru_valid))
                if r2 > r_valid_interp * r_valid_interp:
                    s = r_valid_interp / np.sqrt(r2)
                    pux *= s
                    puy *= s
                    r2 = r_valid_interp * r_valid_interp

            a = (r2 * (k0 + r2 * (k1 + k4 * r2)) + _1) / (r2 * (k5 + r2 * (k6 + k7 * r2)) + _1)
            b = _2_k2 * puy + _2_k3 * pux
            cx = r2 * (k3_p_k8 + k9 * r2)
            cy = r2 * (k2_p_k10 + k11 * r2)
            s = a + b
            dst[i, 0] = pux * s + cx
            dst[i, 1] = puy * s + cy
    else:
        for i in range(pun.shape[0]):
            pux = pun[i, 0]
            puy = pun[i, 1]
            r2 = pux * pux + puy * puy
            a = (r2 * (k0 + r2 * (k1 + k4 * r2)) + _1) / (r2 * (k5 + r2 * (k6 + k7 * r2)) + _1)
            b = _2_k2 * puy + _2_k3 * pux
            cx = r2 * (k3_p_k8 + k9 * r2)
            cy = r2 * (k2_p_k10 + k11 * r2)
            s = a + b
            dst[i, 0] = pux * s + cx
            dst[i, 1] = puy * s + cy

    return dst


@numba.njit(error_model='numpy', cache=False)
def _undistort_points(
    pn,
    d,
    polar_ud_valid,
    check_validity,
    clip_to_valid,
    include_jacobian,
    n_iter_fixed_point,
    n_iter_newton,
    lambda_,
):
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = d
    _1 = np.float32(1)
    _2 = np.float32(2)
    k3_p_k8 = k3 + k8
    k2_p_k10 = k2 + k10
    _2_k2 = _2 * k2
    _2_k3 = _2 * k3

    (ru_valid, tu_valid), (rd_valid, td_valid) = polar_ud_valid
    ru2_valid_min = np.square(np.min(ru_valid))

    lambda_ = np.float32(lambda_)

    pn_valid = pn

    if clip_to_valid:
        pn_valid = np.empty_like(pn)
        # in this case we do process the points that are outside the valid region
        # in this case we scale back the norm to the max valid radius
        # and then apply the distortion model
        rn2_valid_min = np.square(np.min(rd_valid))
        for i in range(pn.shape[0]):
            pnx, pny = pn[i, 0], pn[i, 1]
            rn2 = pnx * pnx + pny * pny
            if rn2 > rn2_valid_min:
                t = np.arctan2(pny, pnx)
                r_interp = np.float32(np.interp(t, td_valid, rd_valid))
                r2_interp = r_interp * r_interp
                if rn2 > r2_interp:
                    scale = r_interp / np.sqrt(rn2)
                    pn_valid[i, 0] = scale * pnx
                    pn_valid[i, 1] = scale * pny
                else:
                    pn_valid[i, 0] = pnx
                    pn_valid[i, 1] = pny
            else:
                pn_valid[i, 0] = pnx
                pn_valid[i, 1] = pny
        all_valid = True
    elif check_validity:
        # in this case we don't process points that are outside the valid region
        # but will set nan to the output for them
        rn2_valid_max = np.square(np.max(rd_valid))
        rn2_valid_min = np.square(np.min(rd_valid))
        i_valid = np.empty(pn.shape[0], dtype=np.int32)
        i_out = 0
        for i in range(pn.shape[0]):
            pnx, pny = pn[i, 0], pn[i, 1]
            rn2 = pnx * pnx + pny * pny
            if rn2 > rn2_valid_max:
                continue
            if rn2 > rn2_valid_min:
                t = np.arctan2(pny, pnx)
                r_interp = np.float32(np.interp(t, td_valid, rd_valid))
                r2_interp = r_interp * r_interp
                if rn2 > r2_interp:
                    continue
            i_valid[i_out] = i
            i_out += 1
        all_valid = i_out == pn.shape[0]
        if not all_valid:
            pn_valid = np.empty((i_out, 2), dtype=np.float32)
            for i in range(i_out):
                pn_valid[i, 0] = pn[i_valid[i], 0]
                pn_valid[i, 1] = pn[i_valid[i], 1]
            i_valid = i_valid[:i_out]

    pun_hat = np.empty_like(pn_valid)
    if include_jacobian:
        jac_hat = np.empty((pn_valid.shape[0], 4), dtype=np.float32)

    # Project pun back to valid region again
    for i in range(pn_valid.shape[0]):
        pux, puy = pn_valid[i, 0], pn_valid[i, 1]
        r2 = pux * pux + puy * puy
        if r2 > ru2_valid_min:
            t = np.arctan2(puy, pux)
            r_interp = np.float32(np.interp(t, tu_valid, ru_valid))
            r2_interp = r_interp * r_interp
            if r2 > r2_interp:
                scale = r_interp / np.sqrt(r2)
                pun_hat[i, 0] = scale * pux
                pun_hat[i, 1] = scale * puy
                continue

        pun_hat[i, 0] = pux
        pun_hat[i, 1] = puy

    # Fixed point iteration
    for _ in range(n_iter_fixed_point):
        for i in range(pn_valid.shape[0]):
            pnx, pny = pn_valid[i, 0], pn_valid[i, 1]
            pux_hat, puy_hat = pun_hat[i, 0], pun_hat[i, 1]
            r2 = pux_hat * pux_hat + puy_hat * puy_hat
            inv_a = (r2 * (k5 + r2 * (k6 + k7 * r2)) + _1) / (r2 * (k0 + r2 * (k1 + k4 * r2)) + _1)
            b = _2_k2 * puy_hat + _2_k3 * pux_hat
            cx = r2 * (k3_p_k8 + k9 * r2)
            cy = r2 * (k2_p_k10 + k11 * r2)
            pun_hat[i, 0] = (pnx - cx - pux_hat * b) * inv_a
            pun_hat[i, 1] = (pny - cy - puy_hat * b) * inv_a

    # Project pun back to valid region again
    for i in range(pn_valid.shape[0]):
        pux, puy = pun_hat[i, 0], pun_hat[i, 1]
        r2 = pux * pux + puy * puy
        if r2 > ru2_valid_min:
            t = np.arctan2(pun_hat[i, 1], pun_hat[i, 0])
            r_interp = np.float32(np.interp(t, tu_valid, ru_valid))
            r2_interp = r_interp * r_interp
            if r2 > r2_interp:
                scale = r_interp / np.sqrt(r2)
                pun_hat[i, 0] = scale * pux
                pun_hat[i, 1] = scale * puy

    # Newton iteration
    if include_jacobian:
        n_iter_newton += 1

    for i_iter in range(n_iter_newton):
        for i in range(pn_valid.shape[0]):
            r2 = pun_hat[i, 0] * pun_hat[i, 0] + pun_hat[i, 1] * pun_hat[i, 1]
            _2_x = _2 * pun_hat[i, 0]
            _2_y = _2 * pun_hat[i, 1]

            k9_r2 = k9 * r2
            k11_r2 = k11 * r2
            k4_r2 = k4 * r2
            k7_r2 = k7 * r2
            x2 = k3_p_k8 + k9_r2
            x16 = k2_p_k10 + k11_r2
            x6 = k1 + k4_r2
            x7 = k0 + r2 * x6
            x10 = k6 + k7_r2
            x11 = k5 + r2 * x10
            x13 = _1 / (r2 * x11 + _1)
            x29 = x13 * (r2 * x7 + _1)
            x14 = pun_hat[i, 0] * _2_k3 + pun_hat[i, 1] * _2_k2 + x29
            x26 = x13 * x13 * ((r2 * (k4_r2 + x6) + x7) - x29 * x29 * (r2 * (k7_r2 + x10) + x11))
            x19 = pun_hat[i, 0] * x26 + k3
            x21 = pun_hat[i, 1] * x26 + k2
            x27 = k9_r2 + x2
            x28 = k11_r2 + x16
            pnx_hat = pun_hat[i, 0] * x14 + r2 * x2
            pny_hat = pun_hat[i, 1] * x14 + r2 * x16

            j00 = _2_x * (x19 + x27) + x14
            j11 = _2_y * (x21 + x28) + x14
            j01 = _2_x * x21 + _2_y * x27
            j10 = _2_y * x19 + _2_x * x28
            j01_times_j10 = j01 * j10
            det = j00 * j11 - j01_times_j10

            if np.fabs(det) < np.float32(0.05):
                j00 += lambda_
                j11 += lambda_
                det = j00 * j11 - j01_times_j10

            inv_det = _1 / det

            if include_jacobian and i_iter == n_iter_newton - 1:
                jac_hat[i, 0] = j11 * inv_det
                jac_hat[i, 1] = -j01 * inv_det
                jac_hat[i, 2] = -j10 * inv_det
                jac_hat[i, 3] = j00 * inv_det
                continue

            err_x = pn_valid[i, 0] - pnx_hat
            err_y = pn_valid[i, 1] - pny_hat
            pun_hat[i, 0] += inv_det * (j11 * err_x - j01 * err_y)
            pun_hat[i, 1] += inv_det * (j00 * err_y - j10 * err_x)

    # now move back to original indices
    if check_validity and not all_valid:
        if include_jacobian:
            # we concat the jacobian elementwise to make a 6-channel output
            punjac_hat_full = np.full((pn.shape[0], 6), fill_value=np.float32(np.nan))
            punjac_hat_full[i_valid, :2] = pun_hat
            punjac_hat_full[i_valid, 2:] = jac_hat
            return punjac_hat_full
        else:
            pun_hat_full = np.full_like(pn, fill_value=np.float32(np.nan))
            pun_hat_full[i_valid, :] = pun_hat
            return pun_hat_full
    else:
        if include_jacobian:
            return np.concatenate((pun_hat, jac_hat), axis=1)
        else:
            return pun_hat


def distort_points_fisheye(pu, d, check_validity=True, clip_to_valid=False, dst=None):
    if check_validity and clip_to_valid:
        raise ValueError("Only one of check_validity and clip_to_valid can be True")

    rud = lensform.validity.fisheye_valid_r_max_cached(d)
    return _distort_points_fisheye(
        pu, d, rud_valid=rud, check_validity=check_validity, clip_to_valid=clip_to_valid, dst=dst
    )


def undistort_points_fisheye(pn, d, check_validity=True):
    rud = lensform.validity.fisheye_valid_r_max_cached(d)
    return _undistort_points_fisheye(pn, d, rud, n_iter_newton=3, check_validity=check_validity)


@numba.njit(error_model='numpy', cache=True)
def _distort_points_fisheye_nodst(pun, d, rud_valid, check_validity, clip_to_valid):
    d0, d1, d2, d3 = d
    _0 = np.float32(0)
    _1 = np.float32(1)
    _nan = np.float32(np.nan)
    ru_valid, rd_valid = rud_valid

    dst = np.empty_like(pun)

    if clip_to_valid:
        for i in range(pun.shape[0]):
            pux_old = pun[i, 0]
            puy_old = pun[i, 1]
            r2 = pux_old * pux_old + puy_old * puy_old
            if r2 == _0:
                dst[i, 0] = _0
                dst[i, 1] = _0
            else:
                r = np.sqrt(r2) if r2 <= ru_valid * ru_valid else ru_valid
                t = np.arctan(r)
                t2 = t * t
                t_d = t * (_1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3))))
                s = t_d / r
                dst[i, 0] = pux_old * s
                dst[i, 1] = puy_old * s
    else:
        for i in range(pun.shape[0]):
            pux_old = pun[i, 0]
            puy_old = pun[i, 1]
            r2 = pux_old * pux_old + puy_old * puy_old
            if check_validity and r2 > ru_valid * ru_valid:
                dst[i, 0] = _nan
                dst[i, 1] = _nan
            elif r2 == _0:
                dst[i, 0] = _0
                dst[i, 1] = _0
            else:
                r = np.sqrt(r2)
                t = np.arctan(r)
                t2 = t * t
                t_d = t * (_1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3))))
                s = t_d / r
                dst[i, 0] = pux_old * s
                dst[i, 1] = puy_old * s
    return dst


@numba.njit(error_model='numpy', cache=True)
def _distort_points_fisheye(pun, d, rud_valid, check_validity, clip_to_valid, dst):
    d0, d1, d2, d3 = d
    _0 = np.float32(0)
    _1 = np.float32(1)
    _nan = np.float32(np.nan)
    ru_valid, rd_valid = rud_valid

    if dst is None:
        dst = np.empty_like(pun)

    if clip_to_valid:
        for i in range(pun.shape[0]):
            pux_old = pun[i, 0]
            puy_old = pun[i, 1]
            r2 = pux_old * pux_old + puy_old * puy_old
            if r2 == _0:
                dst[i, 0] = _0
                dst[i, 1] = _0
            else:
                r = np.sqrt(r2) if r2 <= ru_valid * ru_valid else ru_valid
                t = np.arctan(r)
                t2 = t * t
                t_d = t * (_1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3))))
                s = t_d / r
                dst[i, 0] = pux_old * s
                dst[i, 1] = puy_old * s
    else:
        for i in range(pun.shape[0]):
            pux_old = pun[i, 0]
            puy_old = pun[i, 1]
            r2 = pux_old * pux_old + puy_old * puy_old
            if check_validity and r2 > ru_valid * ru_valid:
                dst[i, 0] = _nan
                dst[i, 1] = _nan
            elif r2 == _0:
                dst[i, 0] = _0
                dst[i, 1] = _0
            else:
                r = np.sqrt(r2)
                t = np.arctan(r)
                t2 = t * t
                t_d = t * (_1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3))))
                s = t_d / r
                dst[i, 0] = pux_old * s
                dst[i, 1] = puy_old * s
    return dst


@numba.njit(error_model='numpy', cache=True)
def _undistort_points_fisheye(pn, d, rud_valid, n_iter_newton, check_validity):
    # Max radius for initial guess in Netwon's method. If we start at the very max (ru_valid)
    # then the optimization does not always converge as there are numerical issues at the very edge.
    ru_valid, rd_valid = rud_valid
    max_initial_t = np.arctan(ru_valid) * 0.95

    # The array t_td contains t and td as the two columns
    # t is theta, td is a 9th degree polynomial of theta, as in the fisheye distortion model
    # td is the radius of the distorted point
    # our goal will be to recover the correct value for t
    # initially we set it to min(td, max_initial_t)
    t_td = np.empty((pn.shape[0], 2), dtype=np.float32)

    if check_validity:
        i_valid = np.empty(pn.shape[0], dtype=np.int32)
        i_out = 0
        for i in range(pn.shape[0]):
            rn2 = pn[i, 0] * pn[i, 0] + pn[i, 1] * pn[i, 1]
            if rn2 <= rd_valid * rd_valid:
                rn = np.sqrt(rn2)
                t_td[i_out, 0] = min(rn, max_initial_t)
                t_td[i_out, 1] = rn
                i_valid[i_out] = i
                i_out += 1
        all_valid = i_out == pn.shape[0]
        i_valid = i_valid[:i_out]
    else:
        for i in range(pn.shape[0]):
            rn = np.sqrt(pn[i, 0] * pn[i, 0] + pn[i, 1] * pn[i, 1])
            t_td[i, 0] = min(rn, max_initial_t)
            t_td[i, 1] = rn

    _0 = np.float32(0)
    _1 = np.float32(1)
    _3 = np.float32(3)
    _5 = np.float32(5)
    _7 = np.float32(7)
    _9 = np.float32(9)
    d0, d1, d2, d3 = d
    _3_d0 = _3 * d0
    _5_d1 = _5 * d1
    _7_d2 = _7 * d2
    _9_d3 = _9 * d3

    # Newton's method to solve for t
    for _ in range(n_iter_newton):
        for i in range(t_td.shape[0]):
            t2 = t_td[i, 0] * t_td[i, 0]
            # the numerator is the current residual
            numer = t_td[i, 0] * (_1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3)))) - t_td[i, 1]
            # the denominator is the derivative, following Newton's method
            denom = _1 + t2 * (_3_d0 + t2 * (_5_d1 + t2 * (_7_d2 + t2 * _9_d3)))
            t_td[i, 0] -= numer / denom

    pun_hat = np.empty((pn.shape[0], 2), dtype=np.float32)
    _nan = np.float32(np.nan)
    if check_validity and not all_valid:
        i_out = 0
        for i_in in range(i_valid.shape[0]):
            i_out_next = i_valid[i_in]
            while i_out < i_out_next:
                pun_hat[i_out, 0] = _nan
                pun_hat[i_out, 1] = _nan
                i_out += 1

            if t_td[i_in, 1] == _0:
                pun_hat[i_out, 0] = _0
                pun_hat[i_out, 1] = _0
            else:
                s = np.tan(t_td[i_in, 0]) / t_td[i_in, 1]
                pun_hat[i_out, 0] = pn[i_out, 0] * s
                pun_hat[i_out, 1] = pn[i_out, 1] * s
            i_out += 1
        while i_out < pn.shape[0]:
            pun_hat[i_out, 0] = _nan
            pun_hat[i_out, 1] = _nan
            i_out += 1
    else:
        pun_hat = np.empty((pn.shape[0], 2), dtype=np.float32)
        for i in range(pn.shape[0]):
            if t_td[i, 1] == _0:
                pun_hat[i, 0] = _0
                pun_hat[i, 1] = _0
            else:
                s = np.tan(t_td[i, 0]) / t_td[i, 1]
                pun_hat[i, 0] = pn[i, 0] * s
                pun_hat[i, 1] = pn[i, 1] * s

    return pun_hat
