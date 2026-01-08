import enum
import functools

import numba
import numpy as np

import deltacamera.core
import deltacamera.coordframes
import deltacamera.distortion
import deltacamera.validity


class LensType(enum.Enum):
    NONE = 0
    USUAL = 1
    FISH = 2


def get_lens_type(camera) -> LensType:
    if not camera.has_distortion():
        return LensType.NONE
    elif camera.has_fisheye_distortion():
        return LensType.FISH
    else:
        return LensType.USUAL


def make(h, w, old_camera, new_camera, precomp):
    old_lens = get_lens_type(old_camera)
    new_lens = get_lens_type(new_camera)

    # NONE
    if old_lens == LensType.NONE and new_lens == LensType.NONE:
        return make_no_distortion(
            h,
            w,
            new_camera.intrinsic_matrix,
            new_camera.R,
            old_camera.R,
            old_camera.intrinsic_matrix,
        )

    # USUAL
    if old_lens == LensType.USUAL and new_lens == LensType.NONE:
        old_d = old_camera.get_distortion_coeffs(12)
        polar_ud_old = deltacamera.validity.get_valid_distortion_region_cached(old_d.tobytes())
        return make_old_distorted(
            h,
            w,
            old_camera.intrinsic_matrix,
            old_d,
            polar_ud_old,
            old_camera.R,
            new_camera.R,
            new_camera.intrinsic_matrix,
        )

    if old_lens == LensType.NONE and new_lens == LensType.USUAL:
        new_d = new_camera.get_distortion_coeffs(12)
        if precomp:
            undist_maps, undist_f = precomp_maps_undistort_cached(new_d)
            return make_new_distorted_from_precomp(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_camera.R,
                new_camera.R,
                undist_maps,
                undist_f,
                new_camera.intrinsic_matrix,
            )
        else:
            polar_ud_new = deltacamera.validity.get_valid_distortion_region_cached(new_d.tobytes())
            return make_new_distorted(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_camera.R,
                new_camera.R,
                new_d,
                polar_ud_new,
                new_camera.intrinsic_matrix,
            )

    if old_lens == LensType.USUAL and new_lens == LensType.USUAL:
        old_d = old_camera.get_distortion_coeffs(12)
        polar_ud_old = deltacamera.validity.get_valid_distortion_region_cached(old_d.tobytes())
        new_d = new_camera.get_distortion_coeffs(12)
        if precomp:
            undist_maps, undist_f = precomp_maps_undistort_cached(new_d)
            return make_both_distorted_from_precomp(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_d,
                polar_ud_old,
                old_camera.R,
                new_camera.R,
                undist_maps,
                undist_f,
                new_camera.intrinsic_matrix,
            )
        else:
            polar_ud_new = deltacamera.validity.get_valid_distortion_region_cached(new_d.tobytes())
            return make_both_distorted(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_d,
                polar_ud_old,
                old_camera.R,
                new_camera.R,
                new_d,
                polar_ud_new,
                new_camera.intrinsic_matrix,
            )

    # FISH
    if old_lens == LensType.FISH and new_lens == LensType.NONE:
        if precomp:
            dist_map, rud_old = precomp_map_distort_fisheye_cached(old_camera.distortion_coeffs)
            return make_old_fisheye_from_precomp(
                h,
                w,
                old_camera.intrinsic_matrix,
                dist_map,
                rud_old,
                old_camera.R,
                new_camera.R,
                new_camera.intrinsic_matrix,
            )
        else:
            rud_old = deltacamera.validity.fisheye_valid_r_max_cached(old_camera.distortion_coeffs)
            return make_old_fisheye(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_camera.distortion_coeffs,
                rud_old,
                old_camera.R,
                new_camera.R,
                new_camera.intrinsic_matrix,
            )

    if old_lens == LensType.NONE and new_lens == LensType.FISH:
        if precomp:
            undist_map, rud_new = precomp_map_undistort_fisheye_cached(
                new_camera.distortion_coeffs
            )
            return make_new_fisheye_from_precomp(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_camera.R,
                new_camera.R,
                undist_map,
                rud_new,
                new_camera.intrinsic_matrix,
            )
        else:
            rud_new = deltacamera.validity.fisheye_valid_r_max_cached(new_camera.distortion_coeffs)
            return make_new_fisheye(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_camera.R,
                new_camera.R,
                new_camera.distortion_coeffs,
                rud_new,
                new_camera.intrinsic_matrix,
            )

    if old_lens == LensType.FISH and new_lens == LensType.FISH:
        if precomp:
            undist_map, rud_new = precomp_map_undistort_fisheye_cached(
                new_camera.distortion_coeffs
            )
            dist_map, rud_old = precomp_map_distort_fisheye_cached(old_camera.distortion_coeffs)
            return make_both_fisheye_from_precomp(
                h,
                w,
                old_camera.intrinsic_matrix,
                dist_map,
                rud_old,
                old_camera.R,
                new_camera.R,
                undist_map,
                rud_new,
                new_camera.intrinsic_matrix,
            )
        else:
            rud_old = deltacamera.validity.fisheye_valid_r_max_cached(old_camera.distortion_coeffs)
            rud_new = deltacamera.validity.fisheye_valid_r_max_cached(new_camera.distortion_coeffs)
            return make_both_fisheye(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_camera.distortion_coeffs,
                rud_old,
                old_camera.R,
                new_camera.R,
                new_camera.distortion_coeffs,
                rud_new,
                new_camera.intrinsic_matrix,
            )

    # MIX
    if old_lens == LensType.USUAL and new_lens == LensType.FISH:
        old_d = old_camera.get_distortion_coeffs(12)
        polar_ud_old = deltacamera.validity.get_valid_distortion_region_cached(old_d.tobytes())
        if precomp:
            undist_map, rud_new = precomp_map_undistort_fisheye_cached(
                new_camera.distortion_coeffs
            )
            return make_old_usual_new_fisheye_from_precomp(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_d,
                polar_ud_old,
                old_camera.R,
                new_camera.R,
                undist_map,
                rud_new,
                new_camera.intrinsic_matrix,
            )
        else:
            rud_new = deltacamera.validity.fisheye_valid_r_max_cached(new_camera.distortion_coeffs)
            return make_old_usual_new_fisheye(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_d,
                polar_ud_old,
                old_camera.R,
                new_camera.R,
                new_camera.distortion_coeffs,
                rud_new,
                new_camera.intrinsic_matrix,
            )

    if old_lens == LensType.FISH and new_lens == LensType.USUAL:
        new_d = new_camera.get_distortion_coeffs(12)
        if precomp:
            undist_maps, undist_f = precomp_maps_undistort_cached(new_d)
            dist_map, rud_old = precomp_map_distort_fisheye_cached(old_camera.distortion_coeffs)
            return make_old_fisheye_new_usual_from_precomp(
                h,
                w,
                old_camera.intrinsic_matrix,
                dist_map,
                rud_old,
                old_camera.R,
                new_camera.R,
                undist_maps,
                undist_f,
                new_camera.intrinsic_matrix,
            )
        else:
            rud_old = deltacamera.validity.fisheye_valid_r_max_cached(old_camera.distortion_coeffs)
            polar_ud_new = deltacamera.validity.get_valid_distortion_region_cached(new_d.tobytes())
            return make_both_dist(
                h,
                w,
                old_camera.intrinsic_matrix,
                old_camera.distortion_coeffs,
                rud_old,
                None,
                old_camera.R,
                new_camera.R,
                new_d,
                None,
                polar_ud_new,
                new_camera.intrinsic_matrix,
            )
            # return make_old_fisheye_new_usual(
            #     h, w,
            #     old_camera.intrinsic_matrix,
            #     old_camera.distortion_coeffs,
            #     rud_old,
            #     old_camera.R,
            #     new_camera.R,
            #     new_d,
            #     polar_ud_new,
            #     new_camera.intrinsic_matrix,
            # )

    raise ValueError(
        f"Unsupported lens type combination: {old_lens.name} -> {new_lens.name}")


@numba.njit(error_model='numpy', cache=True)
def make_no_distortion(h, w, K_old, R_old, R_new, K_new):
    if np.array_equal(R_old, R_new):
        return make_change_intrinsics(h, w, K_old, K_new)
    else:
        H = deltacamera.coordframes.mul_K_M_Kinv(K_old, R_old @ R_new.T, K_new)
        return make_homography(h, w, H)


@numba.njit(error_model='numpy', cache=False)
def make_old_distorted(h, w, K_old, d_old, polar_ud_old, R_old, R_new, K_new):
    pun_old = make_start(h, w, R_old, R_new, K_new)
    pn_old = deltacamera.distortion._distort_points(
        pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_old_fisheye(h, w, K_old, d_old, rud_old, R_old, R_new, K_new):
    pun_old = make_start(h, w, R_old, R_new, K_new)
    pn_old = deltacamera.distortion._distort_points_fisheye(
        pun_old, d_old, rud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_new_distorted(h, w, K_old, R_old, R_new, d_new, polar_ud_new, K_new):
    pn_new = make_undo_intrinsics(h, w, K_new)
    pun_new = deltacamera.distortion._undistort_points(
        pn_new,
        d_new,
        polar_ud_new,
        check_validity=True,
        clip_to_valid=False,
        include_jacobian=False,
        n_iter_fixed_point=5,
        n_iter_newton=2,
        lambda_=5e-1,
    )
    p_old = apply_end_inplace(pun_new, K_old, R_old, R_new)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_both_dist(
    h, w, K_old, d_old, rud_old, polar_ud_old, R_old, R_new, d_new, rud_new, polar_ud_new, K_new
):
    if np.array_equal(d_old, d_new) and np.array_equal(R_old, R_new):
        return make_change_intrinsics(h, w, K_old, K_new)

    pn_new = make_undo_intrinsics(h, w, K_new)
    if polar_ud_new is not None:
        pun_new = deltacamera.distortion._undistort_points(
            pn_new,
            d_new,
            polar_ud_new,
            check_validity=True,
            clip_to_valid=False,
            include_jacobian=False,
            n_iter_fixed_point=5,
            n_iter_newton=2,
            lambda_=5e-1,
        )
    elif rud_new is not None:
        pun_new = deltacamera.distortion._undistort_points_fisheye(
            pn_new, d_new, rud_new, n_iter_newton=3, check_validity=True
        )
    else:
        pun_new = pn_new

    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    if polar_ud_old is not None:
        pn_old = deltacamera.distortion._distort_points(
            pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
        )
    elif rud_old is not None:
        pn_old = deltacamera.distortion._distort_points_fisheye(
            pun_old, d_old, rud_old, check_validity=True, clip_to_valid=False, dst=pun_old
        )
    else:
        pn_old = pun_old
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_both_distorted(
    h, w, K_old, d_old, polar_ud_old, R_old, R_new, d_new, polar_ud_new, K_new
):
    if np.array_equal(d_old, d_new) and np.array_equal(R_old, R_new):
        return make_change_intrinsics(h, w, K_old, K_new)

    pn_new = make_undo_intrinsics(h, w, K_new)
    pun_new = deltacamera.distortion._undistort_points(
        pn_new,
        d_new,
        polar_ud_new,
        check_validity=True,
        clip_to_valid=False,
        include_jacobian=False,
        n_iter_fixed_point=5,
        n_iter_newton=2,
        lambda_=5e-1,
    )
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = deltacamera.distortion._distort_points(
        pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_new_fisheye(h, w, K_old, R_old, R_new, d_new, rud_new, K_new):
    pn_new = make_undo_intrinsics(h, w, K_new)
    pun_new = deltacamera.distortion._undistort_points_fisheye(
        pn_new, d_new, rud_new, n_iter_newton=3, check_validity=True
    )
    p_old = apply_end_inplace(pun_new, K_old, R_old, R_new)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_both_fisheye(h, w, K_old, d_old, rud_old, R_old, R_new, d_new, rud_new, K_new):
    if np.array_equal(d_old, d_new) and np.array_equal(R_old, R_new):
        return make_change_intrinsics(h, w, K_old, K_new)

    pn_new = make_undo_intrinsics(h, w, K_new)
    pun_new = deltacamera.distortion._undistort_points_fisheye(
        pn_new, d_new, rud_new, n_iter_newton=3, check_validity=True
    )
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = deltacamera.distortion._distort_points_fisheye(
        pun_old, d_old, rud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


# Cross-model maps (between the 12-parameter model and fisheye)
@numba.njit(error_model='numpy', cache=True)
def make_old_usual_new_fisheye(
    h, w, K_old, d_old, polar_ud_old, R_old, R_new, d_new, rud_new, K_new
):
    pn_new = make_undo_intrinsics(h, w, K_new)
    pun_new = deltacamera.distortion._undistort_points_fisheye(
        pn_new, d_new, rud_new, n_iter_newton=3, check_validity=True
    )
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = deltacamera.distortion._distort_points(
        pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_old_fisheye_new_usual(
    h, w, K_old, d_old, rud_old, R_old, R_new, d_new, polar_ud_new, K_new
):
    pn_new = make_undo_intrinsics(h, w, K_new)
    pun_new = deltacamera.distortion._undistort_points(
        pn_new,
        d_new,
        polar_ud_new,
        check_validity=True,
        clip_to_valid=False,
        include_jacobian=False,
        n_iter_fixed_point=5,
        n_iter_newton=2,
        lambda_=5e-1,
    )
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = deltacamera.distortion._distort_points_fisheye(
        pun_old, d_old, rud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_start(h, w, R_old, R_new, K_new):
    if np.array_equal(R_old, R_new):
        return make_undo_intrinsics(h, w, K_new)
    else:
        H = deltacamera.coordframes.mul_M_K_inv(R_old @ R_new.T, K_new)
        return make_homography(h, w, H)


@numba.njit(error_model='numpy', cache=True)
def apply_middle_inplace(pun_new, R_old, R_new):
    if np.array_equal(R_old, R_new):
        return pun_new
    else:
        return deltacamera.coordframes.transform_perspective(pun_new, R_old @ R_new.T, dst=pun_new)


@numba.njit(error_model='numpy', cache=True)
def apply_end_inplace(pun_new, K_old, R_old, R_new):
    if np.array_equal(R_old, R_new):
        return deltacamera.coordframes.apply_intrinsics(pun_new, K_old, dst=pun_new)
    else:
        H = deltacamera.coordframes.get_projection_matrix3x3(K_old, R_old @ R_new.T)
        return deltacamera.coordframes.transform_perspective(pun_new, H, dst=pun_new)


@numba.njit(error_model='numpy', cache=True)
def make_intrinsics(h, w, K):
    K00 = K[0, 0]
    K01 = K[0, 1]
    K02 = K[0, 2]
    K11 = K[1, 1]
    K12 = K[1, 2]
    p_new = np.empty((h, w, 2), np.float32)

    for i in range(h):
        py = np.float32(i)
        pny = py * K11 + K12
        offs = py * K01 + K02
        for j in range(w):
            px = np.float32(j)
            p_new[i, j, 0] = px * K00 + offs
            p_new[i, j, 1] = pny
    return p_new.reshape(-1, 2)


@numba.njit(error_model='numpy', cache=True)
def make_undo_intrinsics(h, w, K):
    L = deltacamera.coordframes.inv_intrinsic_matrix(K)
    return make_intrinsics(h, w, L)


@numba.njit(error_model='numpy', cache=True)
def make_homography(h, w, H):
    p_old = np.empty((h, w, 2), np.float32)
    H00, H01, H02, H10, H11, H12, H20, H21, H22 = H.flat
    _0 = np.float32(0)
    _1 = np.float32(1)
    _nan = np.float32(np.nan)
    for i in range(h):
        py_new = np.float32(i)
        b0 = H01 * py_new + H02
        b1 = H11 * py_new + H12
        b2 = H21 * py_new + H22

        for j in range(w):
            px_new = np.float32(j)
            pz_old = H20 * px_new + b2
            if pz_old <= _0:
                p_old[i, j, 0] = _nan
                p_old[i, j, 1] = _nan
            else:
                inv_z = _1 / pz_old
                p_old[i, j, 0] = (H00 * px_new + b0) * inv_z
                p_old[i, j, 1] = (H10 * px_new + b1) * inv_z
    return p_old.reshape(-1, 2)


@numba.njit(error_model='numpy', cache=True)
def make_change_intrinsics(h, w, K_do, K_undo):
    L = deltacamera.coordframes.relative_intrinsics(K_undo, K_do)
    return make_intrinsics(h, w, L)


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_new_distorted_from_precomp(h, w, K_old, R_old, R_new, undist_maps, undist_f, K_new):
    pn_new = make_undo_intrinsics(h, w, K_new)
    pun_new = apply_distortion_map_inplace(pn_new, undist_maps, undist_f)
    p_old = apply_end_inplace(pun_new, K_old, R_old, R_new)
    return p_old


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_both_distorted_from_precomp(
    h, w, K_old, d_old, polar_ud_old, R_old, R_new, undist_maps, undist_f, K_new
):
    pn_new = make_undo_intrinsics(h, w, K_new)
    pun_new = apply_distortion_map_inplace(pn_new, undist_maps, undist_f)
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = deltacamera.distortion._distort_points(
        pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_new_fisheye_from_precomp(h, w, K_old, R_old, R_new, undist_map, rud_new, K_new):
    pn_new = make_undo_intrinsics(h, w, K_new)
    ru_new, rd_new = rud_new
    pun_new = apply_fisheye_map_inplace(pn_new, undist_map, rd_new)
    p_old = apply_end_inplace(pun_new, K_old, R_old, R_new)
    return p_old


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_old_fisheye_from_precomp(h, w, K_old, dist_map, rud_old, R_old, R_new, K_new):
    pun_old = make_start(h, w, R_old, R_new, K_new)
    ru_old, rd_old = rud_old
    pn_old = apply_fisheye_map_inplace(pun_old, dist_map, np.minimum(ru_old, np.float32(3.0)))
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_both_fisheye_from_precomp(
    h, w, K_old, dist_map, rud_old, R_old, R_new, undist_map, rud_new, K_new
):
    pn_new = make_undo_intrinsics(h, w, K_new)
    ru_new, rd_new = rud_new
    pun_new = apply_fisheye_map_inplace(pn_new, undist_map, rd_new)
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    ru_old, rd_old = rud_old
    pn_old = apply_fisheye_map_inplace(pun_old, dist_map, np.minimum(ru_old, np.float32(3.0)))
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_old_usual_new_fisheye_from_precomp(
    h, w, K_old, d_old, polar_ud_old, R_old, R_new, undist_map, rud_new, K_new
):
    pn_new = make_undo_intrinsics(h, w, K_new)
    ru_new, rd_new = rud_new
    pun_new = apply_fisheye_map_inplace(pn_new, undist_map, rd_new)
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = deltacamera.distortion._distort_points(
        pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_old_fisheye_new_usual(
    h, w, K_old, d_old, rud_old, R_old, R_new, d_new, polar_ud_new, K_new
):
    pn_new = make_undo_intrinsics(h, w, K_new)
    pun_new = deltacamera.distortion._undistort_points(
        pn_new,
        d_new,
        polar_ud_new,
        check_validity=True,
        clip_to_valid=False,
        include_jacobian=False,
        n_iter_fixed_point=5,
        n_iter_newton=2,
        lambda_=5e-1,
    )
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = deltacamera.distortion._distort_points_fisheye(
        pun_old, d_old, rud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_old_fisheye_new_usual_from_precomp(
    h, w, K_old, dist_map, rud_old, R_old, R_new, undist_maps, undist_f, K_new
):
    pn_new = make_undo_intrinsics(h, w, K_new)
    pun_new = apply_distortion_map_inplace(pn_new, undist_maps, undist_f)
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    ru_old, rd_old = rud_old
    pn_old = apply_fisheye_map_inplace(pun_old, dist_map, np.minimum(ru_old, np.float32(3.0)))
    p_old = deltacamera.coordframes.apply_intrinsics(pn_old, K_old, dst=pn_old)
    return p_old


def precomp_maps_undistort_cached(distortion_coeffs, fov_degrees_max=120, res=1024):
    return _precomp_maps_undistort(distortion_coeffs.tobytes(), fov_degrees_max, res)


def precomp_map_undistort_fisheye_cached(distortion_coeffs, res=2048):
    return _precomp_map_undistort_fisheye(distortion_coeffs.tobytes(), res)


def precomp_map_distort_fisheye_cached(distortion_coeffs, res=2048):
    return _precomp_map_distort_fisheye(distortion_coeffs.tobytes(), res)


@functools.lru_cache(128)
def _precomp_maps_undistort(distortion_coeffs: bytes, fov_degrees_max: float, res: int):
    polar_ud = deltacamera.validity.get_valid_distortion_region_cached(distortion_coeffs)
    d = np.frombuffer(distortion_coeffs, np.float32)
    return precomp_maps_undistort(d, polar_ud, fov_degrees_max, res)


@functools.lru_cache(128)
def _precomp_map_undistort_fisheye(distortion_coeffs: bytes, res: int):
    rud_valid = deltacamera.validity._fisheye_valid_r_max_cached(distortion_coeffs)
    d = np.frombuffer(distortion_coeffs, np.float32)
    return precomp_map_undistort_fisheye(d, rud_valid, res), rud_valid


@functools.lru_cache(128)
def _precomp_map_distort_fisheye(distortion_coeffs: bytes, res: int):
    rud_valid = deltacamera.validity._fisheye_valid_r_max_cached(distortion_coeffs)
    d = np.frombuffer(distortion_coeffs, np.float32)
    return precomp_map_distort_fisheye(d, rud_valid, res), rud_valid


@numba.njit(error_model='numpy', cache=True)
def precomp_maps_undistort(distortion_coeffs: np.ndarray, polar_ud, fov_degrees_max, res):
    _0 = np.float32(0)
    _1 = np.float32(1)
    _2 = np.float32(2)
    c = np.float32(res - 1) * np.float32(0.5)
    (ru, tu), (rd, td) = polar_ud
    f1 = c / np.max(rd)
    f2 = c / np.tan(np.deg2rad(np.float32(fov_degrees_max)) * np.float32(0.5))
    f = max(f1, f2)
    K = np.array([[f, _0, c], [_0, f, c]], np.float32)
    pn = make_undo_intrinsics(res, res, K)
    punjac = deltacamera.distortion._undistort_points(
        pn,
        distortion_coeffs,
        polar_ud,
        check_validity=True,
        clip_to_valid=False,
        include_jacobian=True,
        n_iter_fixed_point=5,
        n_iter_newton=2,
        lambda_=5e-1,
    )
    # the 6 values are the undistorted points and the jacobian of the undistortion function
    return punjac.reshape(res, res, 6), f


@numba.njit(error_model='numpy', cache=True)
def precomp_map_undistort_fisheye(d: np.ndarray, rud_valid, res):
    ru_valid, rd_valid = rud_valid
    max_initial_t = np.arctan(ru_valid) * np.float32(0.95)
    t_d2 = np.linspace(0, rd_valid * rd_valid, res).astype(np.float32)
    t_d = np.sqrt(t_d2)
    t = np.minimum(t_d, max_initial_t)

    _1 = np.float32(1)
    d0, d1, d2, d3 = d
    _3_d0 = np.float32(3) * d0
    _5_d1 = np.float32(5) * d1
    _7_d2 = np.float32(7) * d2
    _9_d3 = np.float32(9) * d3

    # Newton's method to solve for t
    n_iter_newton = 4
    for _ in range(n_iter_newton):
        for i in range(t.shape[0]):
            t2 = t[i] * t[i]
            numer = t[i] * (_1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3)))) - t_d[i]
            denom = _1 + t2 * (_3_d0 + t2 * (_5_d1 + t2 * (_7_d2 + t2 * _9_d3)))
            t[i] -= numer / denom

    result = np.tan(t) / t_d
    result[0] = _1
    return result


@numba.njit(error_model='numpy', cache=True)
def precomp_map_distort_fisheye(d: np.ndarray, rud_valid, res):
    ru_valid, rd_valid = rud_valid
    ru_practical = np.minimum(ru_valid, np.float32(3.0))
    d0, d1, d2, d3 = d
    _1 = np.float32(1)
    r2 = np.linspace(0, ru_practical * ru_practical, res).astype(np.float32)
    r = np.sqrt(r2)
    t = np.arctan(r)
    t2 = t * t
    t_d = t * (_1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3))))
    result = t_d / r
    result[0] = _1
    return result


@numba.njit(error_model='numpy', cache=True)
def apply_distortion_map_inplace(points, maps, f):
    c = np.float32(maps.shape[1] - 1) * np.float32(0.5)
    c_p_half = np.float32(maps.shape[1]) * np.float32(0.5)
    inv_f = np.float32(1) / f
    _nan = np.float32(np.nan)

    for i in numba.prange(points.shape[0]):
        pnx = points[i, 0]
        pny = points[i, 1]
        map_x = np.floor(pnx * f + c_p_half)
        map_y = np.floor(pny * f + c_p_half)
        imap_x = np.int64(map_x)
        imap_y = np.int64(map_y)

        if map_x >= 0 and map_y >= 0 and imap_x < maps.shape[1] and imap_y < maps.shape[0]:
            pux, puy, j00, j01, j10, j11 = maps[imap_y, imap_x]
            dx = pnx - (map_x - c) * inv_f
            dy = pny - (map_y - c) * inv_f
            pux += dx * j00 + dy * j01
            puy += dx * j10 + dy * j11
            points[i, 0] = pux
            points[i, 1] = puy
        else:
            points[i, 0] = _nan
            points[i, 1] = _nan
    return points


@numba.njit(error_model='numpy', cache=True)
def apply_fisheye_map_inplace(points, map, r_max):
    """Apply the precomputed map that associates points along the radius with scaling factors"""
    # The array `map` has to be indexed according to r**2 and its endpoint corresponds to r_max**2.
    _nan = np.float32(np.nan)
    map_factor_new = np.float32(map.shape[0] - 1) / (r_max * r_max)
    for i in numba.prange(points.shape[0]):
        rn2 = points[i, 0] * points[i, 0] + points[i, 1] * points[i, 1]
        if rn2 < r_max * r_max:
            s = map[np.int64(np.rint(rn2 * map_factor_new))]
            points[i, 0] *= s
            points[i, 1] *= s
        else:
            points[i, 0] = _nan
            points[i, 1] = _nan
    return points
