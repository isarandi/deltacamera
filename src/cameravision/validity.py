import functools
from typing import TYPE_CHECKING

import cameravision.cameravision
import cameravision.coordframes
import cameravision.distortion
import cameravision.reprojection
import cameravision.util
import cv2
import numba
import numpy as np
import shapely
import shapely.ops
from rlemasklib import RLEMask
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

if TYPE_CHECKING:
    from cameravision import Camera


# =============================================================================
# Valid distortion region (Brown-Conrady)
# =============================================================================


def get_valid_distortion_region(
    camera: "Camera",
    limit: float = 5,
    n_vertices: int = 128,
    n_vertices_coarse: int = 24,
    n_steps_line_search: int = 100,
    n_iter_newton: int = 3,
    cartesian: bool = True,
) -> np.ndarray:
    """Returns the region of normalized image space that remains valid after distortion.

    The region is returned as polygon vertices either in polar or cartesian coordinates.

    The function starts from the principal point and expands radially outwards until
    it hits the place where the Jacobian of the distortion function becomes singular.
    That is the point where the Jacobian is no longer invertible, i.e. the one-to-one mapping is
    lost. This happens because the distortion can fold back onto itself, as it's based on
    polynomials that are calibrated (fitted) only within a certain range of image space, and
    they can go haywire outside of that region.

    More precisely, for each :math:`\theta` direction, we search for the smallest :math:`r` s.t.

    .. math::

        \det(J(r\cos\theta, r\sin\theta))` = 0

    where :math:`J` is the 2x2 Jacobian matrix of the distortion function at a point.

    We start with a coarse search for a smaller number of angles, using a line search outwards to
    find for each angle the first point where the Jacobian determinant becomes negative.

    Then, we linearly interpolate the results (in polar coordinates) to a denser sampling, and
    refine the obtained values using Newton's method.

    Args:
        camera: Camera object.
        limit: Maximum radius in normalized image space to search for the valid region.
        n_vertices: Number of vertices in the final polygon.
        n_vertices_coarse: Number of vertices in the coarse sweep.
        n_steps_line_search: Number of steps in the line search.
        n_iter_newton: Number of iterations in the Newton optimization.
        cartesian: If True, return the region in cartesian coordinates, otherwise in polar.

    Returns:
        The valid region in polar or cartesian coordinates, shape `(n_vertices, 2)`.
    """

    if camera.has_fisheye_distortion():
        return get_valid_distortion_region_fisheye(camera, n_vertices, cartesian)

    dist_coeffs = camera.get_distortion_coeffs(12)
    (ru, tu), (rd, td) = get_valid_distortion_region_cached(
        dist_coeffs.tobytes(),
        limit,
        n_vertices,
        n_vertices_coarse,
        n_steps_line_search,
        n_iter_newton,
    )

    if cartesian:
        return polar_to_cartesian(ru[:-1], tu[:-1]), polar_to_cartesian(rd[:-1], td[:-1])
    else:
        return np.stack((ru[:-1], tu[:-1]), axis=-1), np.stack((rd[:-1], td[:-1]), axis=-1)


def get_valid_distortion_region_fisheye(
    camera: "Camera", n_vertices: int = 128, cartesian: bool = True
):
    ru, rd = fisheye_valid_r_max_cached(camera.distortion_coeffs)
    t = np.linspace(-np.pi, np.pi, n_vertices)[:-1]
    if cartesian:
        return polar_to_cartesian_same_radius(ru, t), polar_to_cartesian_same_radius(rd, t)
    else:
        ru = np.repeat(ru, t.shape[0])
        rd = np.repeat(rd, t.shape[0])
        return np.stack((ru, t), axis=-1), np.stack((rd, t), axis=-1)


@functools.lru_cache(128)
def get_valid_distortion_region_cached(
    dist_coeffs: bytes,
    limit: float = 5,
    n_vertices: int = 128,
    n_vertices_coarse: int = 24,
    n_steps_line_search: int = 100,
    n_iter_newton: int = 3,
) -> np.ndarray:
    dist_coeffs = np.frombuffer(dist_coeffs, np.float32)
    return _get_valid_distortion_region_polar(
        dist_coeffs,
        np.float32(limit),
        n_vertices,
        n_vertices_coarse,
        n_steps_line_search,
        n_iter_newton,
    )


def are_points_in_valid_region(points_undist: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Check if points in normalized image space are within the valid region of distortion.

    Args:
        points_undist: Points in normalized image space, shape `(n_points, 2)`.
        d: Distortion coefficients (at most 12), according to OpenCV's order.

    Returns:
        Boolean array of shape `(n_points,)` indicating if the points are within the valid region.
    """
    polar_u, polar_d = get_valid_distortion_region_cached(d.astype(np.float32).tobytes())
    return _are_points_in_polar_region(points_undist, polar_u)


def are_normalized_distorted_points_in_valid_region(
    points: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Check if points in normalized distorted image space are within the valid region of
    distortion.

    Args:
        points: Points in normalized distorted image space, shape `(n_points, 2)`.
        d: Distortion coefficients (at most 12), according to OpenCV's order.

    Returns:
        Boolean array of shape `(n_points,)` indicating if the points are within the valid region.
    """
    polar_u, polar_d = get_valid_distortion_region_cached(d.astype(np.float32).tobytes())
    return _are_points_in_polar_region(points, polar_d)
    #
    # d = pad_axis_to_size(d, 12, -1)
    #
    # (ru, tu), (rd, td) = _get_valid_distortion_region_cached(d.astype(np.float32).tobytes())
    # points_r, points_t = cartesian_to_polar(points)
    # points_r_interp = np.interp(points_t, td, rd)
    # return points_r < points_r_interp
    # return _compute_winding_numbers(region_dist_x, region_dist_y, points[:, 0], points[:, 1]) != 0
    # return shapely.contains_xy(Polygon(region_dist), points)


@numba.njit(error_model='numpy', cache=True)
def _are_points_in_polar_region(
    points: np.ndarray, polar_valid: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    r_points, t_points = cartesian_to_polar(points)
    r_valid, t_valid = polar_valid
    r_interp = np.interp(t_points, t_valid, r_valid).astype(np.float32)
    return r_points <= r_interp


def shape_transform(func, geom):
    def inner(x, y):
        inp = np.stack((x, y), axis=-1).astype(np.float32)
        result = func(inp)
        return result[:, 0], result[:, 1]

    return shapely.ops.transform(inner, geom)


def shape_image_to_undistorted(camera, shape):
    return shape_transform(
        lambda x: camera.image_to_camera(x, validate_distortion=False, depth=None, dst=x), shape
    )


def shape_undistorted_to_image(camera, shape):
    return shape_transform(
        lambda x: camera.camera_to_image(x, validate_distortion=False, dst=x), shape
    )


def shape_apply_intrinsics(camera, shape):
    K = camera.intrinsic_matrix
    return shape_transform(lambda x: cameravision.coordframes.apply_intrinsics(x, K, dst=x), shape)


@numba.njit(error_model='numpy', cache=True)
def _get_valid_distortion_region_polar(
    d, limit, n_bins, n_bins_coarse, n_steps_line_search, n_iter_newton
):
    """Find valid region boundary in polar coordinates via coarse-to-fine search."""
    # Limit search to avoid rational distortion asymptote (denominator zero)
    asymptote_limit = np.sqrt(solve_cubic_smallest_nonneg_root(d[7], d[6], d[5], np.float32(1)))
    limit = np.minimum(limit, asymptote_limit)

    _2_pi = np.float32(2.0 * np.pi)
    _pi = np.float32(np.pi)
    theta = np.linspace(-_pi, _pi, n_bins_coarse)[:-1].astype(np.float32)

    # Coarse line search
    radii = line_search(theta, d, n_steps=n_steps_line_search, limit=limit)
    theta = np.concatenate((theta, theta[:1] + _2_pi), axis=0)
    radii = np.concatenate((radii, radii[:1]), axis=0)

    # Interpolate to dense sampling and refine with Newton
    theta_dense = np.linspace(-_pi, _pi, n_bins)[:-1].astype(np.float32)
    radii_dense = np.interp(theta_dense, theta, radii).astype(np.float32)
    isinf = np.isinf(radii_dense)
    if np.any(isinf):
        radii_dense[isinf] = limit
        radii_dense[~isinf] = newton_optimize(
            radii_dense[~isinf], theta_dense[~isinf], d, n_iter=n_iter_newton
        )
    else:
        radii_dense = newton_optimize(radii_dense, theta_dense, d, n_iter=n_iter_newton)
    radii_dense = np.minimum(radii_dense, limit)
    radii_dense[radii_dense < 0] = 0  # Divergence symptom

    # Convert boundary to distorted space
    pu = polar_to_cartesian(radii_dense, theta_dense)
    pn = cameravision.distortion._distort_points(
        pu, d, polar_ud_valid=None, check_validity=False, clip_to_valid=False, dst=pu
    )
    radii_dense_distorted, theta_dense_distorted = cartesian_to_polar(pn)

    # Sort by angle (distortion may reorder points)
    order_distorted = np.argsort(theta_dense_distorted)
    radii_dense_distorted = radii_dense_distorted[order_distorted]
    theta_dense_distorted = theta_dense_distorted[order_distorted]

    # Wrap around for interpolation continuity
    theta_dense_distorted = np.concatenate(
        (
            theta_dense_distorted[-1:] - _2_pi,
            theta_dense_distorted,
            theta_dense_distorted[:1] + _2_pi,
        ),
        axis=0,
    )
    radii_dense_distorted = np.concatenate(
        (radii_dense_distorted[-1:], radii_dense_distorted, radii_dense_distorted[:1]), axis=0
    )
    theta_dense = np.concatenate(
        (theta_dense[-1:] - _2_pi, theta_dense, theta_dense[:1] + _2_pi), axis=0
    )
    radii_dense = np.concatenate((radii_dense[-1:], radii_dense, radii_dense[:1]), axis=0)
    return (radii_dense, theta_dense), (radii_dense_distorted, theta_dense_distorted)


@numba.njit(error_model='numpy', cache=True)
def polar_to_cartesian(r, t):
    xy = np.empty((r.shape[0], 2), dtype=r.dtype)
    for i in range(r.shape[0]):
        xy[i, 0] = r[i] * np.cos(t[i])
        xy[i, 1] = r[i] * np.sin(t[i])
    return xy


@numba.njit(error_model='numpy', cache=True)
def polar_to_cartesian_same_radius(r, t):
    xy = np.empty((t.shape[0], 2), dtype=t.dtype)
    for i in range(t.shape[0]):
        xy[i, 0] = r * np.cos(t[i])
        xy[i, 1] = r * np.sin(t[i])
    return xy


@numba.njit(error_model='numpy', cache=True)
def cartesian_to_polar(xy):
    r = np.empty((xy.shape[0],), dtype=xy.dtype)
    t = np.empty((xy.shape[0],), dtype=xy.dtype)
    for i in range(xy.shape[0]):
        r[i] = np.hypot(xy[i, 0], xy[i, 1])
        t[i] = np.arctan2(xy[i, 1], xy[i, 0])
    return r, t


@numba.njit(error_model='numpy', cache=True)
def line_search(t, d, n_steps, limit):
    # Spacing is denser near origin where boundary typically lies
    # must use astype because numba's linspace has no dtype argument
    u = np.linspace(np.float32(0), np.float32(1), n_steps).astype(np.float32)
    r = u ** np.float32(1.5) * limit
    jac_det = jacobian_det_polar(r[np.newaxis, :], t[:, np.newaxis], d)
    r_out = np.empty_like(t)
    for i in range(jac_det.shape[0]):
        for j in range(jac_det.shape[1]):
            if jac_det[i, j] < 0:
                r_out[i] = np.float32(0) if j == 0 else r[j - 1]
                break
        else:
            r_out[i] = np.inf
    return r_out


@numba.njit(error_model='numpy', cache=True)
def newton_optimize(r, t, d, n_iter):
    r_new = r.copy()
    for _ in range(n_iter):
        f, df_per_dr = jacobian_det_and_prime_polar(r_new, t, d)
        r_new -= f / df_per_dr
    return r_new


@numba.njit(error_model='numpy', cache=True)
def jacobian_det_polar(r, t, d):
    """Jacobian determinant of the Brown-Conrady distortion at polar coords (r, t).

    Returns det(J) where J = d(x_distorted, y_distorted) / d(x, y).
    Zero crossings mark the boundary where distortion becomes non-invertible.
    """
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = d

    _1 = np.float32(1)
    _2 = np.float32(2)
    _4 = np.float32(4)

    x0 = np.sin(t)
    x1 = np.cos(t)
    x2 = _2 * r
    x3 = x2 * (k2 * x1 - k3 * x0)
    x4 = k2 * x0
    x5 = k3 * x1
    x6 = r * r
    x7 = k1 + k4 * x6
    x8 = k0 + x6 * x7
    x9 = x6 * x8 + _1
    x10 = k6 + k7 * x6
    x11 = k5 + x10 * x6
    x13 = _1 / (x11 * x6 + _1)
    x23 = x4 + x5
    x24 = x13 * x9
    x14 = x24 + x2 * x23
    x15 = x1 * x14
    x16 = k9 * x6
    x17 = k3 + k8
    x18 = _2 * r * x6
    x19 = (
        x13 * (x2 * x8 + x6 * (k4 * x18 + x2 * x7))
        + _2 * x23
        - x13 * x24 * (x11 * x2 + x6 * (k7 * x18 + x10 * x2))
    )
    x20 = x0 * x14
    x21 = k11 * x6
    x22 = k10 + k2
    return (x0 * x3 + x15) * (r * (_4 * x16 + _2 * x17 + x1 * x19) + x15) - (x1 * x3 - x20) * (
        r * (_4 * x21 + _2 * x22 + x0 * x19) + x20
    )


@numba.njit(error_model='numpy', cache=True)
def jacobian_det_and_prime_polar(r, t, d):
    """Jacobian determinant of the distortion function at a point in polar coordinates, and the
    derivative of this w.r.t. r.

    Let us denote the distorted coordinates as :math:`(\tilde{x}, \tilde{y})` and the undistorted
    coordinates as :math:`(x, y)`.

    This function returns :math:`f(r, \theta)` and :math:`\frac{\partial f(r, \theta)}{\partial r}`
    with

    .. math::

        f(r, \theta) = \det\left(\frac{\boldsymbol{\partial}(\tilde{x},\tilde{y})}{\boldsymbol{\partial} (x, y)}
        \bigg|_{(r\cos\theta, r\sin\theta)}\right)

    The roots of :math:`f(r, \theta)` are the points where the distortion function is locally
    non-invertible, hence the boundary of the valid region of distortion.
    """

    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = d
    _1 = np.float32(1)
    _2 = np.float32(2)
    _4 = np.float32(4)
    _6 = np.float32(6)

    x0 = np.sin(t)
    x1 = np.cos(t)
    x2 = k2 * x1
    x3 = k3 * x0
    x4 = _2 * r
    x47 = x2 - x3
    x5 = x4 * x47
    x6 = k2 * x0
    x7 = k3 * x1
    x8 = r * r
    x9 = k4 * x8
    x10 = k1 + x9
    x11 = x10 * x8
    x12 = k0 + x11
    x13 = x12 * x8 + _1
    x14 = k7 * x8
    x15 = k6 + x14
    x16 = x15 * x8
    x17 = k5 + x16
    x19 = _1 / (x17 * x8 + _1)
    x48 = x6 + x7
    x20 = x13 * x19 + x4 * x48
    x21 = x1 * x20
    x22 = x0 * x5 + x21
    x23 = k9 * x8
    x24 = k3 + k8
    x25 = _2 * r * x8
    x26 = k4 * x25 + x10 * x4
    x27 = x26 * x8
    x28 = k7 * x25 + x15 * x4
    x29 = x28 * x8
    x30 = -_2 * r * x17 - x29
    x31 = x19 * x19
    x32 = x13 * x31
    x49 = x32 * x30
    x33 = x19 * (x12 * x4 + x27) + x49 + _2 * x48
    x34 = x1 * x33
    x35 = x0 * x20
    x36 = x1 * x5 - x35
    x37 = k11 * x8
    x38 = k10 + k2
    x39 = x0 * x33
    x40 = _2 * x47
    x41 = _2 * (k10 + k2)
    x42 = _2 * (k3 + k8)
    x43 = _6 * r
    x44 = _2 * x8
    x45 = _4 * r
    x46 = (
        -x49 * (x17 * x45 + _2 * x29) * x19
        + x19 * (_2 * (k0 + x11) + x26 * x45 + x44 * (k1 + _6 * x9))
        + x30 * x31 * (x12 * x45 + _2 * x27)
        - x32 * (_2 * (k5 + x16) + x28 * x45 + x44 * (k6 + _6 * x14))
    )

    fval = x22 * (r * (_4 * x23 + _2 * x24 + x34) + x21) - x36 * (
        r * (_4 * x37 + _2 * x38 + x39) + x35
    )
    deriv = (
        x22 * (r * (k9 * x43 + x1 * x46) + _6 * x23 + _2 * x34 + x42)
        - x36 * (r * (k11 * x43 + x0 * x46) + _6 * x37 + _2 * x39 + x41)
        + (r * (_4 * x23 + x34 + x42) + x21) * (x0 * x40 + x34)
        - (r * (_4 * x37 + x39 + x41) + x35) * (x1 * x40 - x39)
    )

    return fval, deriv


@numba.njit(error_model='numpy', cache=True)
def solve_cubic_smallest_nonneg_root(a, b, c, d):
    """Solve the cubic equation :math:`ax^3 + bx^2 + cx + d = 0` and return the smallest
    non-negative root or inf if there is no non-negative root."""

    _inf = np.float32(np.inf)
    _1 = np.float32(1)
    _2 = np.float32(2)
    _3 = np.float32(3)
    _4 = np.float32(4)
    _9 = np.float32(9)
    _27 = np.float32(27)

    if a == 0 and b == 0:
        x = -d / c if c != 0 else _inf
        return x if x >= 0 else _inf
    elif a == 0:
        D = c * c - _4 * b * d
        if D >= 0:
            D = np.sqrt(D)
            x1 = (-c + D) / (_2 * b)
            x2 = (-c - D) / (_2 * b)
            x1 = x1 if x1 >= 0 else _inf
            x2 = x2 if x2 >= 0 else _inf
            return min(x1, x2)
        else:
            return _inf

    f = ((_3 * c / a) - ((b**_2) / (a**_2))) / _3
    g = ((_2 * (b**_3)) / (a**_3) - (_9 * b * c / (a**_2)) + (_27 * d / a)) / _27
    h = ((g**_2) / _4) + ((f**_3) / _27)

    if f == 0 and g == 0 and h == 0:
        x = -b / (_3 * a)
        return x if x >= 0 else _inf
    elif h <= 0:
        i = np.sqrt(g**_2 / _4 - h)
        j = np.cbrt(i)
        k = np.arccos(-(g / (_2 * i)))
        M = np.cos(k / _3)
        N = np.sqrt(_3) * np.sin(k / _3)
        P = -(b / (_3 * a))
        x1 = P + _2 * j * M
        x2 = P - j * (M + N)
        x3 = P - j * (M - N)
        x1 = x1 if x1 >= 0 else _inf
        x2 = x2 if x2 >= 0 else _inf
        x3 = x3 if x3 >= 0 else _inf
        return min(x1, x2, x3)
    else:
        S = np.cbrt(-(g / _2) + np.sqrt(h))
        U = np.cbrt(-(g / _2) - np.sqrt(h))
        x = (S + U) - (b / (_3 * a))
        return x if x >= 0 else _inf


# =============================================================================
# Shapely polygon helpers
# =============================================================================


def clean_up_polygon(geom):
    """Extract Polygon/MultiPolygon from GeometryCollection after boolean ops."""
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom

    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon(polys)
    return None


def get_valid_poly(camera: "Camera", imshape=None):
    if not camera.has_distortion():
        if imshape is not None:
            return shapely.box(0, 0, imshape[1], imshape[0])

        su_box = shapely.box(-500, 500, 500, -500)
        return shape_transform(camera.camera_to_image, su_box)

    elif camera.has_nonfisheye_distortion():
        pun_valid = get_valid_distortion_region(camera, cartesian=True) * np.float32(0.99)
        pn_valid = cameravision.distortion._distort_points(
            pun_valid,
            camera.get_distortion_coeffs(12),
            polar_ud_valid=None,
            check_validity=False,
            clip_to_valid=False,
            dst=pun_valid,
        )
        p_valid = cameravision.coordframes.apply_intrinsics(
            pn_valid, camera.intrinsic_matrix, dst=pn_valid
        )

        s_valid = shapely.Polygon(p_valid[:-1])
        if imshape is None:
            return clean_up_polygon(shapely.make_valid(s_valid))

        s_box = shapely.box(0, 0, imshape[1], imshape[0])
        s_inters = s_box & shapely.make_valid(s_valid)
        return clean_up_polygon(s_inters)

    else:
        ru_valid, rd_valid = fisheye_valid_r_max_cached(camera.distortion_coeffs)
        sn_valid = shapely.Point(0, 0).buffer(rd_valid, resolution=16)
        s_valid = shape_apply_intrinsics(camera, sn_valid)
        if imshape is None:
            return clean_up_polygon(s_valid)

        s_box = shapely.box(0, 0, imshape[1], imshape[0])
        s_inters = s_box & shapely.make_valid(s_valid)
        return clean_up_polygon(s_inters)


def get_valid_poly_reproj(old_camera, new_camera, imshape_old=None, imshape_new=None):
    near_z = np.float32(1e-3)
    far_z = np.float32(1e6)

    if np.allclose(new_camera.R, old_camera.R) and cameravision.util.allclose_or_nones(
        new_camera.distortion_coeffs, old_camera.distortion_coeffs
    ):
        return get_valid_poly_reproj_affine(old_camera, new_camera, imshape_old, imshape_new)

    if old_camera.has_distortion():
        pu_old_of_old, pn_old_of_old = get_valid_distortion_region(old_camera)

        if imshape_old is not None:
            # su_old_of_old = shapely.Polygon(pu_old_of_old)
            sn_old_of_old = shapely.Polygon(pn_old_of_old)
            s_old_of_old = shape_apply_intrinsics(old_camera, sn_old_of_old)
            f = old_camera.intrinsic_matrix[0, 0]
            s_old_of_old_box = shapely.box(0, 0, imshape_old[1], imshape_old[0])
            s_old_of_old = s_old_of_old_box & shapely.make_valid(s_old_of_old)
            s_old_of_old = shapely.segmentize(s_old_of_old, max_segment_length=0.05 * f)
            su_old_of_old = shape_image_to_undistorted(old_camera, s_old_of_old)
            # su_old_of_old = su_old_of_old & shapely.make_valid(su_old_of_old_cut)
            su_new_of_old = rot_and_clip_shapely_to_front_space(
                su_old_of_old, new_camera.R @ old_camera.R.T, near_z
            )
        else:
            pu_old_of_old_hom = to_homogeneous(pu_old_of_old)
            pu_new_of_old_hom = pu_old_of_old_hom @ (new_camera.R @ old_camera.R.T).T
            pu_new_of_old_hom = clip_polygon_to_z_positive_eps(pu_new_of_old_hom, near_z)
            pu_new_of_old = pu_new_of_old_hom[:, :2] / pu_new_of_old_hom[:, 2:]
            su_new_of_old = shapely.Polygon(pu_new_of_old)
    else:
        if imshape_old is not None:
            h, w = imshape_old
            p_old_of_old = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
            pu_old_of_old_hom = old_camera.image_to_camera(p_old_of_old)
        else:
            pu_old_of_old = (
                np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], np.float32) * far_z
            )
            pu_old_of_old_hom = to_homogeneous(pu_old_of_old)

        pu_new_of_old_hom = pu_old_of_old_hom @ (new_camera.R @ old_camera.R.T).T
        pu_new_of_old_hom = clip_polygon_to_z_positive_eps(pu_new_of_old_hom, near_z)
        pu_new_of_old = pu_new_of_old_hom[:, :2] / pu_new_of_old_hom[:, 2:]
        su_new_of_old = shapely.Polygon(pu_new_of_old)

    if new_camera.has_distortion():
        pu_new_of_new, pn_new_of_new = get_valid_distortion_region(new_camera)
        su_new_of_new = shapely.Polygon(pu_new_of_new)
        su_new_of_inters = shapely.make_valid(su_new_of_old) & shapely.make_valid(su_new_of_new)
        f = new_camera.intrinsic_matrix[0, 0]
        su_new_of_inters = shapely.segmentize(su_new_of_inters, max_segment_length=0.05 * f)
        s_new_of_inters = shape_undistorted_to_image(new_camera, su_new_of_inters)
    else:
        s_new_of_inters = shape_undistorted_to_image(new_camera, su_new_of_old)

    if imshape_new is not None:
        s_new_of_new_box = shapely.box(0, 0, imshape_new[1], imshape_new[0])
        # s_new_of_inters = shapely.clip_by_rect(s_new_of_inters, 0, 0, imshape_new[1], imshape_new[0])
        # s_new_of_inters = shapely.make_valid(s_new_of_inters)
        if new_camera.has_distortion() or old_camera.has_distortion():
            s_new_of_inters = shapely.make_valid(s_new_of_inters)
        s_new_of_inters = s_new_of_new_box & s_new_of_inters
        return clean_up_polygon(s_new_of_inters)
    else:
        if new_camera.has_distortion() or old_camera.has_distortion():
            s_new_of_inters = shapely.make_valid(s_new_of_inters)
        return clean_up_polygon(s_new_of_inters)


def get_valid_poly_reproj_affine(old_camera, new_camera, imshape_old=None, imshape_new=None):
    if old_camera.has_distortion():
        _, pn = get_valid_distortion_region(old_camera)
        sn = shapely.Polygon(pn)
        s_old_of_old = shape_apply_intrinsics(old_camera, sn)

        if imshape_old is not None:
            s_old_of_old_box = shapely.box(0, 0, imshape_old[1], imshape_old[0])
            s_old_of_old = s_old_of_old_box & shapely.make_valid(s_old_of_old)

        s_new_of_new = shape_apply_intrinsics(new_camera, sn)
        if imshape_new is not None:
            s_new_of_new_box = shapely.box(0, 0, imshape_new[1], imshape_new[0])
            s_new_of_new = s_new_of_new_box & shapely.make_valid(s_new_of_new)

        s_new_of_old = shape_transform(
            lambda x: cameravision.coordframes.reproject_intrinsics(
                x, old_camera.intrinsic_matrix, new_camera.intrinsic_matrix
            ),
            s_old_of_old,
        )
        s_new_of_inters = s_new_of_new & shapely.make_valid(s_new_of_old)
    else:
        if imshape_old is not None:
            s_old_of_old = shapely.box(0, 0, imshape_old[1], imshape_old[0])
            s_new_of_old = shape_transform(
                lambda x: cameravision.coordframes.reproject_intrinsics(
                    x, old_camera.intrinsic_matrix, new_camera.intrinsic_matrix
                ),
                s_old_of_old,
            )
            if imshape_new is not None:
                s_new_of_new = shapely.box(0, 0, imshape_new[1], imshape_new[0])
                s_new_of_inters = s_new_of_new & s_new_of_old
            else:
                s_new_of_inters = s_new_of_old
        elif imshape_new is not None:
            s_new_of_new = shapely.box(0, 0, imshape_new[1], imshape_new[0])
            s_new_of_inters = s_new_of_new
        else:
            bignum = 1e9
            s_new_of_inters = shapely.box(-bignum, -bignum, bignum, bignum)
    return clean_up_polygon(s_new_of_inters)


def shapely_to_rle(s, imshape):
    if isinstance(s, shapely.Polygon):
        return RLEMask.from_polygon(np.array(s.exterior.coords), imshape=imshape[:2])
    if isinstance(s, shapely.MultiPolygon):
        if len(s.geoms) == 0:
            return RLEMask.zeros(imshape[:2])
        return RLEMask.union(
            [
                RLEMask.from_polygon(np.array(poly.exterior.coords), imshape=imshape[:2])
                for poly in s.geoms
            ]
        )
    else:
        raise ValueError("Invalid polygon type.")


def to_homogeneous(points):
    # if points.flags.f_contiguous:
    #    return np.concatenate([points, np.ones_like(points[:, :1])], axis=1)
    return np.squeeze(cv2.convertPointsToHomogeneous(points), axis=1)


def get_valid_mask(camera, imshape):
    if not camera.has_distortion():
        return RLEMask.ones(imshape[:2])

    return shapely_to_rle(get_valid_poly(camera, imshape), imshape)


def get_valid_mask_reproj(old_camera, new_camera, imshape_old, imshape_new):
    return shapely_to_rle(
        get_valid_poly_reproj(old_camera, new_camera, imshape_old, imshape_new), imshape_new
    )


@numba.njit(error_model='numpy', cache=True)
def clip_polygon_to_z_positive_eps(polygon, eps):
    """
    Clips a polygon to the z > eps half-space

    Args:
        polygon (numpy.ndarray): An (N, 3) array where each row is a vertex [x, y, z], dtype float32.
        eps (float): The minimum z-value threshold for clipping, dtype float32.

    Returns:
        numpy.ndarray: A new polygon clipped to z > eps, dtype float32.
    """
    max_vertices = polygon.shape[0] * 2  # Worst-case: doubling the vertices
    clipped_polygon = np.empty((max_vertices, 3), dtype=np.float32)
    used_vertices = 0  # Counter for actually used vertices

    # last_exit_point = None

    num_vertices = polygon.shape[0]
    for i in range(num_vertices):
        v_start = polygon[i]
        v_end = polygon[(i + 1) % num_vertices]

        z_start, z_end = v_start[2], v_end[2]

        # Case 1: Both vertices are in the positive space (z > eps)
        if z_start > eps and z_end > eps:
            clipped_polygon[used_vertices] = v_start
            used_vertices += 1

        # Case 2: Transition from positive to negative (z_start > eps, z_end <= eps)
        elif z_start > eps >= z_end:
            t = (eps - z_start) / (z_end - z_start)
            intersection = v_start + t * (v_end - v_start)
            intersection[2] = eps  # Clamp to the plane z = eps
            clipped_polygon[used_vertices] = v_start
            used_vertices += 1
            clipped_polygon[used_vertices] = intersection
            used_vertices += 1
            # last_exit_point = intersection

        # Case 3: Transition from negative to positive (z_start <= eps, z_end > eps)
        elif z_start <= eps < z_end:
            t = (eps - z_start) / (z_end - z_start)
            intersection = v_start + t * (v_end - v_start)
            intersection[2] = eps  # Clamp to the plane z = eps
            # if last_exit_point is not None:
            #    clipped_polygon[used_vertices] = last_exit_point
            #    used_vertices += 1
            clipped_polygon[used_vertices] = intersection
            used_vertices += 1
            # last_exit_point = None  # Reset exit point after reconnecting

        # Case 4: Both vertices are in the negative space (z <= eps)
        # Do nothing for these edges

    if used_vertices != 0 and not np.all(clipped_polygon[0] == clipped_polygon[used_vertices - 1]):
        clipped_polygon[used_vertices] = clipped_polygon[0]
        used_vertices += 1

    return clipped_polygon[:used_vertices]


def rot_and_clip_shapely_to_front_space(polygon, R, eps=1e-3):
    polygon = clean_up_polygon(polygon)
    eps = np.float32(eps)

    if isinstance(polygon, shapely.Polygon):
        poly_np = np.array(polygon.exterior.coords)
        poly_np_rot_hom = to_homogeneous(poly_np) @ R.T
        poly_np_rot_hom_clipped = clip_polygon_to_z_positive_eps(poly_np_rot_hom, eps)
        poly_np_rot_clipped = poly_np_rot_hom_clipped[:, :2] / poly_np_rot_hom_clipped[:, 2:]
        return shapely.Polygon(poly_np_rot_clipped)
    elif isinstance(polygon, shapely.MultiPolygon):
        return shapely.MultiPolygon(
            [rot_and_clip_shapely_to_front_space(poly, R) for poly in polygon.geoms]
        )
    else:
        raise ValueError("Invalid polygon type.")


# =============================================================================
# Valid distortion region (fisheye / Kannala-Brandt)
# =============================================================================


def fisheye_valid_r_max_cached(d):
    return _fisheye_valid_r_max_cached(d.astype(np.float32).tobytes())


@functools.lru_cache(128)
def _fisheye_valid_r_max_cached(d):
    return fisheye_valid_r_max(np.frombuffer(d, np.float32))


@numba.njit(error_model='numpy', cache=True)
def fisheye_valid_r_max(d):
    """Find the maximum valid radius for fisheye distortion.

    Returns (ru, rd) where ru is the max undistorted radius and rd is the corresponding
    distorted radius. Beyond this, the distortion mapping is no longer monotonic.
    """
    n_steps = 32
    d0, d1, d2, d3 = d
    _0 = np.float32(0)
    _1 = np.float32(1)
    _hpi = np.float32(np.pi / 2)
    _3_d0 = d0 * np.float32(3)
    _5_d1 = d1 * np.float32(5)
    _7_d2 = d2 * np.float32(7)
    _9_d3 = d3 * np.float32(9)

    # Line search: find where derivative of distortion first becomes negative
    t = np.float32(0.1)
    step = (_hpi - t) / n_steps
    for i in range(n_steps):
        t2 = t * t
        deriv = _1 + t2 * (_3_d0 + t2 * (_5_d1 + t2 * (_7_d2 + t2 * _9_d3)))
        if deriv < _0:
            t = _0 if i == 0 else t - step
            break
        t += step
    else:
        # Derivative never became negative, valid up to pi/2
        t = _hpi
        t2 = t * t
        rd = t * (_1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3))))
        return np.float32(np.inf), rd

    # Newton refinement: find exact zero of derivative
    _6_d0 = d0 * np.float32(6)
    _20_d1 = d1 * np.float32(20)
    _42_d2 = d2 * np.float32(42)
    _72_d3 = d3 * np.float32(72)

    for i in range(5):
        t2 = t * t
        deriv = _1 + t2 * (_3_d0 + t2 * (_5_d1 + t2 * (_7_d2 + t2 * _9_d3)))
        deriv2 = t * (_6_d0 + t2 * (_20_d1 + t2 * (_42_d2 + t2 * _72_d3)))
        t -= deriv / deriv2

    t2 = t * t
    ru = np.tan(t)
    rd = t * (_1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3))))
    return ru, rd


def get_optimal_undistorted_intrinsics(
    old_camera: "Camera", old_imshape, new_imshape, alpha, center_principal_point=False
):
    """Compute optimal intrinsics for undistorting an image, similar to cv2.getOptimalNewCameraMatrix.

    Args:
        old_camera: Original distorted camera.
        old_imshape: Shape of original image (h, w).
        new_imshape: Desired output shape (h, w), or None for automatic.
        alpha: Balance between keeping all pixels (1) vs no black borders (0).
        center_principal_point: If True, keep principal point at image center.

    Returns:
        Tuple of (new_intrinsic_matrix, crop_box, valid_polygon).
    """
    # Create undistorted camera with square pixels and no skew
    new_camera = old_camera.copy()
    new_camera.distortion_coeffs = None
    new_camera.square_pixels()
    new_camera.intrinsic_matrix[0, 1] = 0

    # Get valid region polygon and its bounding box (outer_box)
    valid_poly: shapely.Polygon = cameravision.validity.get_valid_poly_reproj(
        old_camera, new_camera, old_imshape, imshape_new=None
    )
    x1, y1, x2, y2 = valid_poly.bounds
    outer_box = np.array([x1, y1, x2 - x1, y2 - y1], np.float32)

    if center_principal_point:
        principal_point = new_camera.intrinsic_matrix[:2, 2] - outer_box[:2]
        rendered_principal_point = np.ceil(principal_point)
        delta = rendered_principal_point - new_camera.intrinsic_matrix[:2, 2]
    else:
        delta = -outer_box[:2]

    # Find inner_box (largest valid rect) and outer_box (full valid region bounds)
    valid_poly = shapely.affinity.translate(valid_poly, xoff=delta[0], yoff=delta[1])
    valid_mask = RLEMask.from_polygon(
        np.array(valid_poly.exterior.coords), (outer_box[3], outer_box[2])
    )
    if new_imshape is not None:
        aspect_ratio = new_imshape[1] / new_imshape[0]
        if center_principal_point:
            inner_box = valid_mask.largest_interior_rectangle_around(
                rendered_principal_point, aspect_ratio
            )
            # Expand outer_box to be centered on principal point
            pp = new_camera.intrinsic_matrix[:2, 2]
            new_width = max(pp[0] - outer_box[0], outer_box[0] + outer_box[2] - pp[0]) * 2
            new_height = max(pp[1] - outer_box[1], outer_box[1] + outer_box[3] - pp[1]) * 2
            outer_box[0] = pp[0] - new_width / 2
            outer_box[1] = pp[1] - new_height / 2
            outer_box[2] = new_width
            outer_box[3] = new_height
        else:
            inner_box = valid_mask.largest_interior_rectangle(aspect_ratio).astype(np.float32)

        # Expand outer_box to match target aspect ratio
        box_aspect = outer_box[2] / outer_box[3]
        if box_aspect < aspect_ratio:
            new_width = outer_box[3] * aspect_ratio
            outer_box[0] -= (new_width - outer_box[2]) / 2
            outer_box[2] = new_width
        else:
            new_height = outer_box[2] / aspect_ratio
            outer_box[1] -= (new_height - outer_box[3]) / 2
            outer_box[3] = new_height
    else:
        if center_principal_point:
            inner_box = valid_mask.largest_interior_rectangle_around(rendered_principal_point)
        else:
            inner_box = valid_mask.largest_interior_rectangle().astype(np.float32)

    # Blend between inner (no black borders) and outer (keep all pixels) based on alpha
    inner_box[:2] -= delta
    box = inner_box * (1 - alpha) + outer_box * alpha

    # Scale to target resolution
    factor = new_imshape[1] / box[2] if new_imshape is not None else 1
    intr_before_shift = new_camera.intrinsic_matrix.copy()
    new_camera.shift_image(-box[:2])
    if new_imshape is not None:
        new_camera.scale_output(factor)
        if center_principal_point:
            new_camera.center_principal_point(new_imshape)

    # Transform boxes and polygon to final output coordinates
    shift = new_camera.intrinsic_matrix[:2, 2] - factor * intr_before_shift[:2, 2]
    inner_box[:] *= factor
    outer_box[:] *= factor
    box[:] *= factor
    inner_box[:2] += shift
    outer_box[:2] += shift
    box[:2] += shift
    valid_poly = shapely.affinity.translate(valid_poly, xoff=-delta[0], yoff=-delta[1])
    valid_poly = shapely.affinity.affine_transform(
        valid_poly, [factor, 0, 0, factor, shift[0], shift[1]]
    )
    return new_camera.intrinsic_matrix, box, valid_poly
