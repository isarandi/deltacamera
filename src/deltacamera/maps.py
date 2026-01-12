import functools

import msgpack_numpy
import numba
import numpy as np
import rlemasklib

from . import core, coordframes, distortion, maps_impl, validity


def make_maps(old_camera, new_camera, output_imshape, precomp_undist_maps):
    p_old = maps_impl.make(
        output_imshape[0],
        output_imshape[1],
        old_camera,
        new_camera,
        precomp_undist_maps,
    )
    return p_old.reshape(output_imshape[0], output_imshape[1], 2)


def get_maps_and_mask(
    old_camera, new_camera, input_imshape, output_imshape, cache=True, precomp_undist_maps=False
):
    if cache:
        return get_maps_and_mask_cached(
            old_camera, new_camera, input_imshape, output_imshape, precomp_undist_maps
        )
    else:
        return make_maps_and_mask(
            old_camera, new_camera, input_imshape, output_imshape, precomp_undist_maps
        )


def get_maps_and_mask_cached(
    old_camera, new_camera, input_imshape, output_imshape, precomp_undist_maps
):
    # Make them relative, so that if next time both are rotated by the same amount,
    # we can reuse the maps. Specifically when the two R are equal.
    # In other cases, if we're not having exactly equal Rs, there will be small numerical errors
    # even if the intended relative rotation is the same. This could be handled in the future.
    # Perhaps through quantization or custom cache with binning then verifying.
    old_camera2 = old_camera.copy()
    new_camera2 = new_camera.copy()
    new_camera2.R = old_camera.R.T @ new_camera.R
    old_camera2.R = np.eye(3, dtype=np.float32)
    old_camdict = cam2dict(old_camera2)
    new_camdict = cam2dict(new_camera2)
    return _get_maps_and_mask_cached(
        old_camdict, new_camdict, tuple(input_imshape), tuple(output_imshape), precomp_undist_maps
    )


@functools.lru_cache(128)
def _get_maps_and_mask_cached(
    old_camdict: bytes,
    new_camdict: bytes,
    input_imshape: tuple[int],
    output_imshape: tuple[int],
    precomp_undist_maps: bool = False,
):
    old_camera = dict2cam(old_camdict)
    new_camera = dict2cam(new_camdict)
    return make_maps_and_mask(
        old_camera, new_camera, input_imshape, output_imshape, precomp_undist_maps
    )


def make_maps_and_mask(old_camera, new_camera, input_imshape, output_imshape, precomp_undist_maps):
    maps = make_maps(old_camera, new_camera, output_imshape, precomp_undist_maps)
    is_valid_cnts_C = is_map_valid_rle_C(maps, imshape_old=input_imshape)
    is_valid_rle = rlemasklib.RLEMask.from_counts(
        is_valid_cnts_C, output_imshape, order='C', validate_sum=True
    )
    return maps, is_valid_rle


def cam2dict(camera):
    dicti = dict(K=camera.intrinsic_matrix, R=camera.R, t=camera.t, d=camera.distortion_coeffs)
    return msgpack_numpy.packb(dicti)


def dict2cam(dicti):
    dicti = msgpack_numpy.unpackb(dicti)
    return core.Camera(
        intrinsic_matrix=dicti["K"],
        rot_world_to_cam=dicti["R"],
        optical_center=dicti["t"],
        distortion_coeffs=dicti["d"],
    )


@numba.njit(error_model='numpy', cache=True)
def is_map_valid_rle_C(map, imshape_old):
    # Return a mask as runlength-encoded counts, in C (row-major) order for whether
    # the points in the map are valid (i.e. coords not NaN and within the old image shape).
    # Foreground is valid, background is invalid.
    if map.shape[0] == 0 or map.shape[1] == 0:
        return np.empty(0, np.uint32)

    v_prev = False
    cnt = 0
    i_out = 0
    rle_counts = np.empty(map.shape[0] * map.shape[1] + 1, np.uint32)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            x = map[i, j, 0]
            y = map[i, j, 1]
            v = 0 <= x < imshape_old[1] and 0 <= y < imshape_old[0]
            if v != v_prev:
                rle_counts[i_out] = cnt
                i_out += 1
                cnt = 0
                v_prev = v
            cnt += 1

    rle_counts[i_out] = cnt
    i_out += 1
    return rle_counts[:i_out]
