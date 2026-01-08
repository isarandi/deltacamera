import functools
import typing

import boxlib
import cv2
import numba
import numpy as np
import rlemasklib

import deltacamera.coordframes
import deltacamera.distortion
import deltacamera.maps
import deltacamera.points_impl
import deltacamera.util
import deltacamera.validity

if typing.TYPE_CHECKING:
    from deltacamera import Camera


def reproject_image_points(
    points,
    old_camera: "Camera",
    new_camera: "Camera",
    precomp_undist_maps: bool = False,
) -> np.ndarray:
    """Reproject 2D image points from `old_camera` to `new_camera`.

    Args:
        points: The 2D image points in the `old_camera` image. Shape (..., 2).
        old_camera: The camera that captured the original points.
        new_camera: The camera to which the points should be reprojected.
        precomp_undist_maps: Whether to precompute undistortion maps for the cameras

    Returns:
        The reprojected 2D image points in the `new_camera` image. Shape (..., 2).
    """
    points = np.asarray(points, dtype=np.float32)
    points_resh = np.ascontiguousarray(points.reshape(-1, 2))
    # The argument order has to be new_camera, old_camera
    # it is because the point and the map implementation is kept analogous
    # but one goes from old to new, the other from new to old when warping
    reproj_resh = deltacamera.points_impl.make(
        points_resh, new_camera, old_camera, precomp_undist_maps
    )
    return reproj_resh.reshape(points.shape)


def reproject_image(
    image: np.ndarray,
    old_camera: "Camera",
    new_camera: "Camera",
    output_imshape: tuple,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value=0,
    interp=None,
    antialias_factor=1,
    dst=None,
    cache_maps=False,
    precomp_undist_maps=True,
    use_linear_srgb=False,
    return_validity_mask=False,
) -> np.ndarray:
    """Transform an `image` captured with `old_camera` to look like it was captured by
    `new_camera`. The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output.
    Ignores the issue of aliasing altogether.

    There are two caching options. If `cache_maps` is True, the coordinate maps for
    this particular reprojection will be cached. If multiple images will be reprojected
    with the same `old_camera` and `new_camera`, it is recommended to set `cache_maps` to True.

    The second option is `precomp_undist_maps`. This is only relevant if the `new_camera` has
    distortion. If `precomp_undist_maps` is True, an undistortion map of that camera's
    distortion coefficients will be precomputed and cached. This precomputed map only depends on
    the distortion coefficients of the new camera, therefore it can be reused in more contexts.
    Therefore, if multiple images will be reprojected with the same `new_camera` distortion
    coefficients, but with varying other parameters, it is recommended to set `precomp_undist_maps`
    to True.

    Args:
        image: the input image
        old_camera: the camera that captured `image`
        new_camera: the camera that should capture the newly returned image
        output_imshape: (height, width) for the output image
        border_mode: OpenCV border mode for treating pixels outside `image`
        border_value: OpenCV border value for treating pixels outside `image`
        interp: OpenCV interpolation to be used for resampling.
        antialias_factor: If larger than 1, first render a higher resolution output image
            that is `antialias_factor` times larger than `output_imshape` and subsequently resize
            it by 'area' interpolation to the desired size.
        dst: destination array (optional)
        cache_maps: Whether to cache the coordinate maps used for reprojection.
        precomp_undist_maps: Whether to precompute and cache undistortion maps for the cameras.
        use_linear_srgb: If True, decode `image` from 8-bit encoded sRGB to 16-bit linear space
            before reprojecting and encode the result back to 8-bit sRGB. This ensures
            correct color interpolation for sRGB inputs.

    Returns:
        The new image.
    """
    if use_linear_srgb:
        image = decode_srgb(image, dst=None) if use_linear_srgb else image

    if antialias_factor == 1:
        result, mask = reproject_image_aliased(
            image,
            old_camera,
            new_camera,
            output_imshape,
            border_mode,
            border_value,
            interp,
            None if use_linear_srgb else dst,
            cache_maps,
            precomp_undist_maps,
        )
        if use_linear_srgb:
            result = encode_srgb(result, dst=dst)

        if return_validity_mask:
            return result, mask
        else:
            return result

    a = antialias_factor
    highres_new_camera = new_camera.scale_output(a, inplace=False)
    highres_new_camera.intrinsic_matrix[:2, 2] += (a - 1) / 2
    highres_imshape = (a * output_imshape[0], a * output_imshape[1])
    highres_result, highres_mask = reproject_image_aliased(
        image,
        old_camera,
        highres_new_camera,
        highres_imshape,
        border_mode,
        border_value,
        interp,
        cache_maps=cache_maps,
        precomp_undist_maps=precomp_undist_maps,
    )
    result = cv2.resize(
        highres_result,
        dsize=(output_imshape[1], output_imshape[0]),
        interpolation=cv2.INTER_AREA,
        dst=None if use_linear_srgb else dst,
    )

    if use_linear_srgb:
        result = encode_srgb(result, dst=dst)

    if return_validity_mask:
        mask = highres_mask.avg_pool2d_valid(kernel_size=(a, a), stride=(a, a))
        return result, mask
    else:
        return result


def reproject_image_aliased(
    image: np.ndarray,
    old_camera: "Camera",
    new_camera: "Camera",
    output_imshape,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0,
    interp=None,
    dst=None,
    cache_maps=False,
    precomp_undist_maps=True,
) -> np.ndarray:
    """Transform an `image` captured with `old_camera` to look like it was captured by
    `new_camera`. The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output.
    Aliasing issues are ignored.
    """
    if interp is None:
        interp = cv2.INTER_LINEAR

    if not np.allclose(old_camera.t, new_camera.t):
        raise Exception(
            "The optical center of the camera must not change, else warping is not enough!"
        )

    if not cache_maps and not old_camera.has_distortion() and not new_camera.has_distortion():
        # No distortion, we can use a perspective warp
        remapped = reproject_image_fast(
            image, old_camera, new_camera, output_imshape, border_mode, border_value, interp, dst
        )
        is_valid_rle = deltacamera.validity.get_valid_mask_reproj(
            old_camera, new_camera, imshape_old=image.shape[:2], imshape_new=output_imshape
        )
        mask_image_by_rle(remapped, ~is_valid_rle, border_value)
        return remapped, is_valid_rle

    if (
        not cache_maps
        and np.allclose(new_camera.R, old_camera.R)
        and deltacamera.util.allclose_or_nones(
            new_camera.distortion_coeffs, old_camera.distortion_coeffs
        )
    ):
        # Only the intrinsics have changed we can use an affine warp
        remapped = reproject_image_affine(
            image, old_camera, new_camera, output_imshape, border_mode, border_value, interp, dst
        )
        is_valid_rle = deltacamera.validity.get_valid_mask_reproj(
            old_camera, new_camera, imshape_old=image.shape[:2], imshape_new=output_imshape
        )
        mask_image_by_rle(remapped, ~is_valid_rle, border_value)
        return remapped, is_valid_rle

    maps, is_valid_rle = deltacamera.maps.get_maps_and_mask(
        old_camera,
        new_camera,
        image.shape[:2],
        output_imshape,
        cache=cache_maps,
        precomp_undist_maps=precomp_undist_maps,
    )
    remapped = cv2.remap(
        image, maps, None, interp, borderMode=border_mode, borderValue=border_value, dst=dst
    )
    if remapped.ndim < image.ndim:
        remapped = np.expand_dims(remapped, -1)
    return remapped, is_valid_rle


@numba.njit(error_model='numpy', cache=True)
def _mask_flat_by_rle(values, rle_counts, value):
    r = 0
    for i_count, cnt in enumerate(rle_counts):
        end = r + cnt
        if i_count % 2 == 1:
            values[r:end] = value
        r = end


@numba.njit(error_model='numpy', cache=True)
def _mask_image_by_rle(im, rle_counts, value):
    r = 0
    for i_count, cnt in enumerate(rle_counts):
        if i_count % 2 == 1:
            for _ in range(cnt):
                i = r % im.shape[0]
                j = r // im.shape[0]
                im[i, j] = value
                r += 1
        else:
            r += cnt


def mask_image_by_rle(im, rle, value):
    """All foreground pixels of image `im` are set to `value`."""
    if im.flags.c_contiguous:
        values = im.reshape(-1)
        n_channels = im.shape[-1] if im.ndim == 3 else 1
        transposed_counts = rle.transpose().counts
        _mask_flat_by_rle(values, transposed_counts * np.int32(n_channels), value)
    elif im.ndim == 2 and im.flags.f_contiguous:
        values = im.reshape(-1, order="F")
        _mask_flat_by_rle(values, rle.counts, value)
    else:
        _mask_image_by_rle(im, rle.counts, value)


def reproject_box(old_box, old_camera, new_camera):
    """Reprojects a bounding box from one camera to another.

    This is an ambiguous operation in general, as the computing the box loses information about
    the precise segmentation of the object.

    Therefore, reproject(bbox(mask)) != bbox(reproject(mask)) in general.
    We apply a heuristic here:

    reproject(box) = (bbox(reproject(corners(box))) + bbox(reproject(side_midpoints(box)))) / 2

    Args:
        old_box: The bounding box in the old camera image.
        old_camera: The camera that captured the image where the object has bbox old_box.
        new_camera: The camera to which the box should be reprojected.

    Returns:
        The reprojected bounding box in the new camera image.
    """
    return (
        reproject_box_corners(old_box, old_camera, new_camera)
        + reproject_box_side_midpoints(old_box, old_camera, new_camera)
    ) / 2


def reproject_box_corners(old_box, old_camera, new_camera):
    """Reprojects a bounding box from one camera to another using its corners."""
    old_corners = boxlib.corners(old_box)
    new_corners = reproject_image_points(old_corners, old_camera, new_camera)
    return boxlib.bb_of_points(new_corners)


def reproject_box_side_midpoints(old_box, old_camera, new_camera):
    """Reprojects a bounding box from one camera to another using its side midpoints."""
    old_side_midpoints = boxlib.side_midpoints(old_box)
    new_side_midpoints = reproject_image_points(old_side_midpoints, old_camera, new_camera)
    return boxlib.bb_of_points(new_side_midpoints)


def reproject_box_inscribed_ellipse(old_box, old_camera, new_camera):
    """Reprojects a bounding box from one camera to another using its inscribed ellipse."""
    old_ellipse_points = boxlib.inscribed_ellipse_points(old_box, n_angles=64, n_radii=16)
    new_ellipse_points = reproject_image_points(old_ellipse_points, old_camera, new_camera)
    return boxlib.bb_of_points(new_ellipse_points)


def reproject_image_fast(
    image,
    old_camera,
    new_camera,
    output_imshape,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=None,
    interp=cv2.INTER_LINEAR,
    dst=None,
):
    """Like reproject_image, but assumes there are no lens distortions."""
    homography = deltacamera.coordframes.mul_K_M_Kinv(
        old_camera.intrinsic_matrix, old_camera.R @ new_camera.R.T, new_camera.intrinsic_matrix
    )

    if border_value is None:
        border_value = 0

    remapped = cv2.warpPerspective(
        image,
        homography,
        (output_imshape[1], output_imshape[0]),
        flags=interp | cv2.WARP_INVERSE_MAP,
        borderMode=border_mode,
        borderValue=border_value,
        dst=dst,
    )

    if remapped.ndim < image.ndim:
        return np.expand_dims(remapped, -1)
    return remapped


def reproject_image_affine(
    image,
    old_camera,
    new_camera,
    output_imshape,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0,
    interp=cv2.INTER_LINEAR,
    dst=None,
):
    K_new = new_camera.intrinsic_matrix
    K_old = old_camera.intrinsic_matrix
    affine_mat_2x3 = deltacamera.coordframes.relative_intrinsics(K_new, K_old)[:2]
    remapped = cv2.warpAffine(
        image,
        affine_mat_2x3,
        (output_imshape[1], output_imshape[0]),
        flags=cv2.WARP_INVERSE_MAP | interp,
        borderMode=border_mode,
        borderValue=border_value,
        dst=dst,
    )
    if remapped.ndim < image.ndim:
        return np.expand_dims(remapped, -1)
    return remapped


def reproject_mask(
    mask,
    old_camera,
    new_camera,
    dst_shape,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0,
    interp=None,
    antialias_factor=1,
    dst=None,
    return_validity_mask=False,
):
    """Reproject a binary mask from one camera view to another.

    Args:
        mask: Binary mask array (bool or uint8) in the old camera's image space.
        old_camera: Source camera.
        new_camera: Target camera.
        dst_shape: Output shape (height, width).
        border_mode: OpenCV border mode for out-of-bounds pixels.
        border_value: Value for border pixels.
        interp: Interpolation method.
        antialias_factor: Antialiasing factor (1 = no antialiasing).
        dst: Optional pre-allocated output array.
        return_validity_mask: If True, also return a mask of valid pixels.

    Returns:
        Reprojected mask, or tuple of (mask, validity_mask) if return_validity_mask=True.
    """
    input_bool = mask.dtype == bool
    if input_bool:
        mask = mask.view(np.uint8)
    mask = np.ascontiguousarray(mask)
    mask = threshold_uint8(mask, 0, 255, dst=None)
    new_mask, mask_mask = reproject_image(
        mask,
        old_camera,
        new_camera,
        dst_shape,
        border_mode,
        border_value,
        interp,
        antialias_factor,
        dst,
        return_validity_mask=True,
    )
    result = threshold_uint8(new_mask, 127, 1, dst=new_mask)
    if input_bool:
        result = result.view(bool)

    if return_validity_mask:
        return result, mask_mask
    else:
        return result


@numba.njit(error_model='numpy', cache=True)
def threshold_uint8(src, thresh, maxval, dst):
    """Equivalent to OpenCV's cv2.threshold with cv2.THRESH_BINARY.

    Equivalent to `cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY, dst)`.

    Equivalent to `dst[:] = np.where(src > thresh, maxval, 0)`.
    """
    src_flat = src.reshape(-1)
    maxval = np.uint8(maxval)
    thresh = np.uint8(thresh)
    if dst is src:
        for i in range(src_flat.shape[0]):
            src_flat[i] = maxval if src_flat[i] > thresh else 0
        return src
    else:
        if dst is None:
            dst = np.empty_like(src, dtype=np.uint8)
        dst_flat = dst.reshape(-1)
        for i in range(src_flat.shape[0]):
            dst_flat[i] = maxval if src_flat[i] > thresh else 0
        return dst


def reproject_rle_mask(
    rle_mask,
    old_camera,
    new_camera,
    dst_shape,
    interp=None,
    antialias_factor=1,
    dst=None,
    precomp_undist_maps=True,
    warp_in_rle=False,
):
    """Reproject an RLE-encoded mask from one camera view to another.

    Args:
        rle_mask: RLE-encoded binary mask in the old camera's image space.
        old_camera: Source camera.
        new_camera: Target camera.
        dst_shape: Output shape (height, width).
        interp: Interpolation method.
        antialias_factor: Antialiasing factor (1 = no antialiasing).
        dst: Optional pre-allocated output array.
        precomp_undist_maps: Whether to precompute undistortion maps.
        warp_in_rle: If True, warp directly in RLE space without decoding (experimental).

    Returns:
        Reprojected RLE mask.
    """
    if (
        warp_in_rle
        and not old_camera.has_fisheye_distortion()
        and not new_camera.has_fisheye_distortion()
    ):
        return _reproject_rle_mask_in_rle(rle_mask, old_camera, new_camera, dst_shape)
    else:
        cropped_rle, bbox = rle_mask.tight_crop()
        old_camera_shifted = old_camera.shift_image(-bbox[:2], inplace=False)
        mask = cropped_rle.to_array(255, order='C')
        new_mask = reproject_image(
            mask,
            old_camera_shifted,
            new_camera,
            dst_shape,
            interp=interp,
            antialias_factor=antialias_factor,
            dst=dst,
            cache_maps=False,
            precomp_undist_maps=precomp_undist_maps,
            use_linear_srgb=False,
        )
        return rlemasklib.RLEMask.from_array(
            new_mask, thresh128=True, is_sparse=rle_mask.density < 0.04
        )


def _reproject_rle_mask_in_rle(rle_mask, old_camera, new_camera, dst_shape):
    valid_rle = deltacamera.validity.get_valid_mask_reproj(
        new_camera, old_camera, None, rle_mask.shape
    )
    rle_masked_to_valid = rle_mask & valid_rle

    if not old_camera.has_distortion() and not new_camera.has_distortion():
        homography = deltacamera.coordframes.mul_K_M_Kinv(
            new_camera.intrinsic_matrix,
            new_camera.R @ old_camera.R.T,
            old_camera.intrinsic_matrix,
        )
        return rle_masked_to_valid.warp_perspective(homography, dst_shape)

    old_d = old_camera.get_distortion_coeffs(12)
    new_d = new_camera.get_distortion_coeffs(12)

    if np.allclose(new_camera.R, old_camera.R) and np.allclose(old_d, new_d):
        return rle_masked_to_valid.warp_affine(
            deltacamera.coordframes.relative_intrinsics(
                new_camera.intrinsic_matrix, old_camera.intrinsic_matrix
            ),
            dst_shape,
        )

    polar_ud1 = deltacamera.validity.get_valid_distortion_region_cached(old_d.tobytes())
    polar_ud2 = deltacamera.validity.get_valid_distortion_region_cached(new_d.tobytes())
    return rle_masked_to_valid.warp_distorted(
        old_camera.R,
        new_camera.R,
        old_camera.intrinsic_matrix,
        new_camera.intrinsic_matrix,
        old_d,
        new_d,
        polar_ud1,
        polar_ud2,
        dst_shape,
    )


@functools.lru_cache
@numba.njit(error_model='numpy', cache=True)
def get_srgb_decoder_lut():
    lut = np.zeros(256, np.float64)
    for i in numba.prange(256):
        x = i / 255
        if x <= 0.04045:
            lut[i] = x / 12.92
        else:
            lut[i] = ((x + 0.055) / 1.055) ** 2.4
        if lut[i] < 0:
            lut[i] = 0
        elif lut[i] > 1:
            lut[i] = 1
    return (lut * ((1 << 16) - 1)).astype(np.uint16)


@functools.lru_cache
@numba.njit(error_model='numpy', cache=True)
def get_srgb_encoder_lut():
    lut = np.zeros(1 << 16, np.float64)
    for i in numba.prange(1 << 16):
        x = i / ((1 << 16) - 1)
        if x <= 0.0031308:
            lut[i] = x * 12.92
        else:
            lut[i] = 1.055 * x ** (1 / 2.4) - 0.055
        if lut[i] < 0:
            lut[i] = 0
        elif lut[i] > 1:
            lut[i] = 1
    return (lut * 255).astype(np.uint8)


@numba.njit(error_model='numpy', cache=True)
def LUT(im, lut, dst):
    out = np.empty(im.shape, lut.dtype) if dst is None else dst
    im_flat = np.ascontiguousarray(im).reshape(-1)
    out_flat = out.reshape(-1)
    for i in numba.prange(im_flat.shape[0]):
        out_flat[i] = lut[im_flat[i]]
    return out


def encode_srgb(im, dst=None):
    """Encodes a linear 16-bit image to 8-bit sRGB.

    Args:
        im: Input pixel values of dtype np.uint16
        dst: Optional destination array of dtype np.uint8

    Returns:
        The sRGB encoded image of dtype np.uint8
    """
    if dst is not None and dst.dtype != np.uint8:
        raise ValueError("The destination dtype must be np.uint8")
    if not im.dtype == np.uint16:
        raise ValueError("The input dtype must be np.uint16")
    if dst is not None and im.size != dst.size:
        raise ValueError("The input and destination arrays must have the same size")

    return LUT(im, get_srgb_encoder_lut(), dst)


def decode_srgb(im, dst=None):
    """Decodes an 8-bit sRGB image to linear 16-bit.

    Args:
        im: Input pixel values of dtype np.uint8
        dst: Optional destination array of dtype np.uint16

    Returns:
        The linear decoded image of dtype np.uint16
    """
    if dst is not None and dst.dtype != np.uint16:
        raise ValueError("The destination dtype must be np.uint16")
    if not im.dtype == np.uint8:
        raise ValueError("The input dtype must be np.uint8")
    if dst is not None and im.size != dst.size:
        raise ValueError("The input and destination arrays must have the same size")

    return LUT(im, get_srgb_decoder_lut(), dst)
