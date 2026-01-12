import numba
import numpy as np

from . import core, coordframes, distortion, validity
from .maps_impl import (LensType, apply_distortion_map_inplace, apply_end_inplace,
                        apply_fisheye_map_inplace, apply_middle_inplace, get_lens_type,
                        precomp_map_distort_fisheye_cached,
                        precomp_map_undistort_fisheye_cached,
                        precomp_maps_undistort_cached)


def make(p, old_camera, new_camera, precomp):
    old_lens = get_lens_type(old_camera)
    new_lens = get_lens_type(new_camera)

    # NONE
    if old_lens == LensType.NONE and new_lens == LensType.NONE:
        return make_no_distortion(
            p,
            old_camera.intrinsic_matrix,
            old_camera.R,
            new_camera.R,
            new_camera.intrinsic_matrix,
        )

    # USUAL
    if old_lens == LensType.USUAL and new_lens == LensType.NONE:
        old_d = old_camera.get_distortion_coeffs(14)
        polar_ud_old = validity.get_valid_distortion_region_cached(old_d.tobytes())
        return make_old_distorted(
            p,
            old_camera.intrinsic_matrix,
            old_d,
            polar_ud_old,
            old_camera.R,
            new_camera.R,
            new_camera.intrinsic_matrix,
        )

    if old_lens == LensType.NONE and new_lens == LensType.USUAL:
        new_d = new_camera.get_distortion_coeffs(14)
        if precomp:
            undist_maps, undist_f = precomp_maps_undistort_cached(new_d)
            return make_new_distorted_from_precomp(
                p,
                old_camera.intrinsic_matrix,
                old_camera.R,
                new_camera.R,
                undist_maps,
                undist_f,
                new_camera.intrinsic_matrix,
            )
        else:
            polar_ud_new = validity.get_valid_distortion_region_cached(new_d.tobytes())
            return make_new_distorted(
                p,
                old_camera.intrinsic_matrix,
                old_camera.R,
                new_camera.R,
                new_d,
                polar_ud_new,
                new_camera.intrinsic_matrix,
            )

    if old_lens == LensType.USUAL and new_lens == LensType.USUAL:
        old_d = old_camera.get_distortion_coeffs(14)
        polar_ud_old = validity.get_valid_distortion_region_cached(old_d.tobytes())
        new_d = new_camera.get_distortion_coeffs(14)
        if precomp:
            undist_maps, undist_f = precomp_maps_undistort_cached(new_d)
            return make_both_distorted_from_precomp(
                p,
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
            polar_ud_new = validity.get_valid_distortion_region_cached(new_d.tobytes())
            return make_both_distorted(
                p,
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
                p,
                old_camera.intrinsic_matrix,
                dist_map,
                rud_old,
                old_camera.R,
                new_camera.R,
                new_camera.intrinsic_matrix,
            )
        else:
            rud_old = validity.fisheye_valid_r_max_cached(old_camera.distortion_coeffs)
            return make_old_fisheye(
                p,
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
                p,
                old_camera.intrinsic_matrix,
                old_camera.R,
                new_camera.R,
                undist_map,
                rud_new,
                new_camera.intrinsic_matrix,
            )
        else:
            rud_new = validity.fisheye_valid_r_max_cached(new_camera.distortion_coeffs)
            return make_new_fisheye(
                p,
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
                p,
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
            rud_old = validity.fisheye_valid_r_max_cached(old_camera.distortion_coeffs)
            rud_new = validity.fisheye_valid_r_max_cached(new_camera.distortion_coeffs)
            return make_both_fisheye(
                p,
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
        old_d = old_camera.get_distortion_coeffs(14)
        polar_ud_old = validity.get_valid_distortion_region_cached(old_d.tobytes())
        if precomp:
            undist_map, rud_new = precomp_map_undistort_fisheye_cached(
                new_camera.distortion_coeffs
            )
            return make_old_usual_new_fisheye_from_precomp(
                p,
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
            rud_new = validity.fisheye_valid_r_max_cached(new_camera.distortion_coeffs)
            return make_old_usual_new_fisheye(
                p,
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
        new_d = new_camera.get_distortion_coeffs(14)
        if precomp:
            undist_maps, undist_f = precomp_maps_undistort_cached(new_d)
            dist_map, rud_old = precomp_map_distort_fisheye_cached(old_camera.distortion_coeffs)
            return make_old_fisheye_new_usual_from_precomp(
                p,
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
            rud_old = validity.fisheye_valid_r_max_cached(old_camera.distortion_coeffs)
            polar_ud_new = validity.get_valid_distortion_region_cached(new_d.tobytes())
            return make_old_fisheye_new_usual(
                p,
                old_camera.intrinsic_matrix,
                old_camera.distortion_coeffs,
                rud_old,
                old_camera.R,
                new_camera.R,
                new_d,
                polar_ud_new,
                new_camera.intrinsic_matrix,
            )


@numba.njit(error_model='numpy', cache=True)
def make_no_distortion(p, K_old, R_old, R_new, K_new):
    if np.array_equal(R_old, R_new):
        return make_change_intrinsics(p, K_old, K_new)
    else:
        H = coordframes.mul_K_M_Kinv(K_old, R_old @ R_new.T, K_new)
        return coordframes.transform_perspective(p, H, dst=None)


@numba.njit(error_model='numpy', cache=False)
def make_old_distorted(p, K_old, d_old, polar_ud_old, R_old, R_new, K_new):
    pun_old = make_start(p, R_old, R_new, K_new)
    pn_old = distortion._distort_points(
        pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = make_intrinsics(pn_old, K_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_old_fisheye(p, K_old, d_old, rud_old, R_old, R_new, K_new):
    pun_old = make_start(p, R_old, R_new, K_new)
    pn_old = distortion._distort_points_fisheye(
        pun_old, d_old, rud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = make_intrinsics(pn_old, K_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_new_distorted(p, K_old, R_old, R_new, d_new, polar_ud_new, K_new):
    pn_new = make_undo_intrinsics(p, K_new)
    pun_new = distortion._undistort_points(
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
def make_both_distorted(p, K_old, d_old, polar_ud_old, R_old, R_new, d_new, polar_ud_new, K_new):
    if np.array_equal(d_old, d_new) and np.array_equal(R_old, R_new):
        return make_change_intrinsics(p, K_old, K_new)

    pn_new = make_undo_intrinsics(p, K_new)
    pun_new = distortion._undistort_points(
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
    pn_old = distortion._distort_points(
        pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = make_intrinsics(pn_old, K_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_new_fisheye(p, K_old, R_old, R_new, d_new, rud_new, K_new):
    pn_new = make_undo_intrinsics(p, K_new)
    pun_new = distortion._undistort_points_fisheye(
        pn_new, d_new, rud_new, n_iter_newton=3, check_validity=True
    )
    p_old = apply_end_inplace(pun_new, K_old, R_old, R_new)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_both_fisheye(p, K_old, d_old, rud_old, R_old, R_new, d_new, rud_new, K_new):
    if np.array_equal(d_old, d_new) and np.array_equal(R_old, R_new):
        return make_change_intrinsics(p, K_old, K_new)

    pn_new = make_undo_intrinsics(p, K_new)
    pun_new = distortion._undistort_points_fisheye(
        pn_new, d_new, rud_new, n_iter_newton=3, check_validity=True
    )
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = distortion._distort_points_fisheye(
        pun_old, d_old, rud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = make_intrinsics(pn_old, K_old)
    return p_old


# Cross-model maps (between the 14-parameter model and fisheye)
@numba.njit(error_model='numpy', cache=True)
def make_old_usual_new_fisheye(p, K_old, d_old, polar_ud_old, R_old, R_new, d_new, rud_new, K_new):
    pn_new = make_undo_intrinsics(p, K_new)
    pun_new = distortion._undistort_points_fisheye(
        pn_new, d_new, rud_new, n_iter_newton=3, check_validity=True
    )
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = distortion._distort_points(
        pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = make_intrinsics(pn_old, K_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_old_fisheye_new_usual(p, K_old, d_old, rud_old, R_old, R_new, d_new, polar_ud_new, K_new):
    pn_new = make_undo_intrinsics(p, K_new)
    pun_new = distortion._undistort_points(
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
    pn_old = distortion._distort_points_fisheye(
        pun_old, d_old, rud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = make_intrinsics(pn_old, K_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_start(p, R_old, R_new, K_new):
    if np.array_equal(R_old, R_new):
        return make_undo_intrinsics(p, K_new)
    else:
        H = coordframes.mul_M_K_inv(R_old @ R_new.T, K_new).astype(np.float32)
        return coordframes.transform_perspective(p, H, dst=None).astype(np.float32)


@numba.njit(error_model='numpy', cache=True)
def make_intrinsics(p, K):
    return coordframes.apply_intrinsics(p, K, dst=None).astype(np.float32)


@numba.njit(error_model='numpy', cache=True)
def make_undo_intrinsics(p, K):
    return coordframes.undo_intrinsics(p, K, dst=None).astype(np.float32)


@numba.njit(error_model='numpy', cache=True)
def make_change_intrinsics(p, K_do, K_undo):
    L = coordframes.relative_intrinsics(K_undo, K_do)
    return coordframes.apply_intrinsics(p, L, dst=None).astype(np.float32)


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_new_distorted_from_precomp(p, K_old, R_old, R_new, undist_maps, undist_f, K_new):
    pn_new = make_undo_intrinsics(p, K_new)
    pun_new = apply_distortion_map_inplace(pn_new, undist_maps, undist_f)
    p_old = apply_end_inplace(pun_new, K_old, R_old, R_new)
    return p_old


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_both_distorted_from_precomp(
    p, K_old, d_old, polar_ud_old, R_old, R_new, undist_maps, undist_f, K_new
):
    pn_new = make_undo_intrinsics(p, K_new)
    pun_new = apply_distortion_map_inplace(pn_new, undist_maps, undist_f)
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = distortion._distort_points(
        pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = make_intrinsics(pn_old, K_old)
    return p_old


@numba.njit(error_model='numpy', cache=True)
def make_new_fisheye_from_precomp(p, K_old, R_old, R_new, undist_map, rud_new, K_new):
    pn_new = make_undo_intrinsics(p, K_new)
    ru_new, rd_new = rud_new
    pun_new = apply_fisheye_map_inplace(pn_new, undist_map, rd_new)
    p_old = apply_end_inplace(pun_new, K_old, R_old, R_new)
    return p_old


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_old_fisheye_from_precomp(p, K_old, dist_map, rud_old, R_old, R_new, K_new):
    pun_old = make_start(p, R_old, R_new, K_new)
    ru_old, rd_old = rud_old
    pn_old = apply_fisheye_map_inplace(pun_old, dist_map, ru_old)
    p_old = make_intrinsics(pn_old, K_old)
    return p_old


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_both_fisheye_from_precomp(
    p, K_old, dist_map, rud_old, R_old, R_new, undist_map, rud_new, K_new
):
    pn_new = make_undo_intrinsics(p, K_new)
    ru_new, rd_new = rud_new
    pun_new = apply_fisheye_map_inplace(pn_new, undist_map, rd_new)
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    ru_old, rd_old = rud_old
    pn_old = apply_fisheye_map_inplace(pun_old, dist_map, ru_old)
    p_old = make_intrinsics(pn_old, K_old)
    return p_old


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_old_usual_new_fisheye_from_precomp(
    p, K_old, d_old, polar_ud_old, R_old, R_new, undist_map, rud_new, K_new
):
    pn_new = make_undo_intrinsics(p, K_new)
    ru_new, rd_new = rud_new
    pun_new = apply_fisheye_map_inplace(pn_new, undist_map, rd_new)
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    pn_old = distortion._distort_points(
        pun_old, d_old, polar_ud_old, check_validity=True, clip_to_valid=False, dst=pun_old
    )
    p_old = make_intrinsics(pn_old, K_old)
    return p_old


@numba.njit(error_model='numpy', cache=True, parallel=False)
def make_old_fisheye_new_usual_from_precomp(
    p, K_old, dist_map, rud_old, R_old, R_new, undist_maps, undist_f, K_new
):
    pn_new = make_undo_intrinsics(p, K_new)
    pun_new = apply_distortion_map_inplace(pn_new, undist_maps, undist_f)
    pun_old = apply_middle_inplace(pun_new, R_old, R_new)
    ru_old, rd_old = rud_old
    pn_old = apply_fisheye_map_inplace(pun_old, dist_map, ru_old)
    p_old = make_intrinsics(pn_old, K_old)
    return p_old
