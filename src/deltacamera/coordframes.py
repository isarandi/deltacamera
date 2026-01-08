import numpy as np
import numba


@numba.njit(error_model='numpy', cache=True)
def project(points, dst):
    n_points = points.shape[0]
    if dst is None:
        dst = np.empty((n_points, 2), dtype=points.dtype)

    for i in numba.prange(n_points):
        z = points[i, 2]
        if z <= 0:
            dst[i, 0] = np.nan
            dst[i, 1] = np.nan
        else:
            inv_z = np.float32(1) / z
            dst[i, 0] = points[i, 0] * inv_z
            dst[i, 1] = points[i, 1] * inv_z
    return dst


@numba.njit(error_model='numpy', cache=True)
def backproject_K_deptharr(points, K, depth_arr, dst):
    n_points = points.shape[0]
    if dst is None:
        dst = np.empty((n_points, 3), dtype=points.dtype)

    K01 = K[0, 1]
    K02 = K[0, 2]
    K12 = K[1, 2]
    _1 = np.float32(1)
    K00_inv = _1 / K[0, 0]
    K11_inv = _1 / K[1, 1]
    for i in numba.prange(n_points):
        y_n = (points[i, 1] - K12) * K11_inv
        dst[i, 0] = (points[i, 0] - K02 - y_n * K01) * K00_inv * depth_arr[i]
        dst[i, 1] = y_n * depth_arr[i]
        dst[i, 2] = depth_arr[i]
    return dst


@numba.njit(error_model='numpy', cache=True)
def backproject_K_depthval(points, K, depth_val, dst):
    n_points = points.shape[0]
    if dst is None:
        dst = np.empty((n_points, 3), dtype=points.dtype)
    depth_per_k11 = depth_val / K[1, 1]
    depth_per_k00 = depth_val / K[0, 0]
    K01 = K[0, 1]
    K02 = K[0, 2]
    K12 = K[1, 2]
    for i in numba.prange(n_points):
        y_n = (points[i, 1] - K12)
        dst[i, 0] = (points[i, 0] - K02 - y_n * K01) * depth_per_k00
        dst[i, 1] = y_n * depth_per_k11
        dst[i, 2] = depth_val
    return dst


@numba.njit(error_model='numpy', cache=True)
def backproject_depthval(points, depth_val, dst):
    n_points = points.shape[0]
    if dst is None:
        dst = np.empty((n_points, 3), dtype=points.dtype)
    for i in numba.prange(n_points):
        dst[i, 0] = points[i, 0] * depth_val
        dst[i, 1] = points[i, 1] * depth_val
        dst[i, 2] = depth_val
    return dst


@numba.njit(error_model='numpy', cache=True)
def backproject_homogeneous(points, dst):
    n_points = points.shape[0]
    if dst is None:
        dst = np.empty((n_points, 3), dtype=points.dtype)
    for i in numba.prange(n_points):
        dst[i, 0] = points[i, 0]
        dst[i, 1] = points[i, 1]
        dst[i, 2] = np.float32(1)
    return dst


@numba.njit(error_model='numpy', cache=True)
def backproject_deptharr(points, depth_arr, dst):
    n_points = points.shape[0]
    if dst is None:
        dst = np.empty((n_points, 3), dtype=points.dtype)
    for i in numba.prange(n_points):
        dst[i, 0] = points[i, 0] * depth_arr[i]
        dst[i, 1] = points[i, 1] * depth_arr[i]
        dst[i, 2] = depth_arr[i]
    return dst


@numba.njit(error_model='numpy', cache=True)
def undo_intrinsics(points, K, dst):
    n_points = points.shape[0]
    if dst is None:
        dst = np.empty_like(points)
    K01 = K[0, 1]
    K02 = K[0, 2]
    K12 = K[1, 2]
    _1 = np.float32(1)
    K00_inv = _1 / K[0, 0]
    K11_inv = _1 / K[1, 1]
    for i in numba.prange(n_points):
        y_out = (points[i, 1] - K12) * K11_inv
        dst[i, 0] = (points[i, 0] - K02 - y_out * K01) * K00_inv
        dst[i, 1] = y_out
    return dst


@numba.njit(error_model='numpy', cache=False)
def apply_intrinsics(points, K, dst):
    n_points = points.shape[0]
    if dst is None:
        dst = np.empty_like(points)

    K00 = K[0, 0]
    K01 = K[0, 1]
    K02 = K[0, 2]
    K11 = K[1, 1]
    K12 = K[1, 2]
    for i in numba.prange(n_points):
        pnx_old = points[i, 0]
        pny_old = points[i, 1]
        dst[i, 0] = K00 * pnx_old + K01 * pny_old + K02
        dst[i, 1] = K11 * pny_old + K12
    return dst


@numba.njit(error_model='numpy', cache=False)
def apply_intrinsics_inplace(points, K):
    n_points = points.shape[0]
    K00 = K[0, 0]
    K01 = K[0, 1]
    K02 = K[0, 2]
    K11 = K[1, 1]
    K12 = K[1, 2]
    for i in numba.prange(n_points):
        pnx_old = points[i, 0]
        pny_old = points[i, 1]
        points[i, 0] = K00 * pnx_old + K01 * pny_old + K02
        points[i, 1] = K11 * pny_old + K12
    return points


@numba.njit(error_model='numpy', cache=True)
def project_and_apply_intrinsics(points, K, dst):
    n_points = points.shape[0]
    if dst is None:
        dst = np.empty((n_points, 2), dtype=points.dtype)
    K00 = K[0, 0]
    K01 = K[0, 1]
    K02 = K[0, 2]
    K11 = K[1, 1]
    K12 = K[1, 2]
    _0 = np.float32(0)
    _1 = np.float32(1)
    _nan = np.float32(np.nan)
    for i in numba.prange(n_points):
        z = points[i, 2]
        if z <= _0:
            dst[i, 0] = _nan
            dst[i, 1] = _nan
        else:
            inv_z = _1 / z
            dst[i, 0] = K00 * points[i, 0] * inv_z + K01 * points[i, 1] * inv_z + K02
            dst[i, 1] = K11 * points[i, 1] * inv_z + K12
    return dst


@numba.njit(error_model='numpy', cache=True)
def get_projection_matrix3x3(K, R):
    K00 = K[0, 0]
    K01 = K[0, 1]
    K02 = K[0, 2]
    K11 = K[1, 1]
    K12 = K[1, 2]
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = R.flat
    M00 = K00 * R00 + K01 * R10 + K02 * R20
    M01 = K00 * R01 + K01 * R11 + K02 * R21
    M02 = K00 * R02 + K01 * R12 + K02 * R22
    M10 = K11 * R10 + K12 * R20
    M11 = K11 * R11 + K12 * R21
    M12 = K11 * R12 + K12 * R22
    return np.array([[M00, M01, M02], [M10, M11, M12], [R20, R21, R22]], np.float32)


@numba.njit(error_model='numpy', cache=True)
def get_projection_matrix3x4(K, R, camloc):
    K00 = K[0, 0]
    K01 = K[0, 1]
    K02 = K[0, 2]
    K11 = K[1, 1]
    K12 = K[1, 2]
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = R.flat
    t0, t1, t2 = camloc
    M00 = K00 * R00 + K01 * R10 + K02 * R20
    M01 = K00 * R01 + K01 * R11 + K02 * R21
    M02 = K00 * R02 + K01 * R12 + K02 * R22
    M10 = K11 * R10 + K12 * R20
    M11 = K11 * R11 + K12 * R21
    M12 = K11 * R12 + K12 * R22
    M03 = -t0 * M00 - t1 * M01 - t2 * M02
    M13 = -t0 * M10 - t1 * M11 - t2 * M12
    M23 = -t0 * R20 - t1 * R21 - t2 * R22
    return np.array([[M00, M01, M02, M03], [M10, M11, M12, M13], [R20, R21, R22, M23]], np.float32)


@numba.njit(error_model='numpy', cache=True)
def get_extrinsic_matrix(R, camloc):
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = R.flat
    t0, t1, t2 = camloc
    _0 = np.float32(0)
    _1 = np.float32(1)
    return np.array(
        [
            [R00, R01, R02, -t0 * R00 - t1 * R01 - t2 * R02],
            [R10, R11, R12, -t0 * R10 - t1 * R11 - t2 * R12],
            [R20, R21, R22, -t0 * R20 - t1 * R21 - t2 * R22],
            [_0, _0, _0, _1],
        ],
        np.float32,
    )


@numba.njit(error_model='numpy', cache=True)
def get_inv_extrinsic_matrix(R, camloc):
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = R.flat
    t0, t1, t2 = camloc
    _0 = np.float32(0)
    _1 = np.float32(1)
    return np.array(
        [[R00, R10, R20, t0], [R01, R11, R21, t1], [R02, R12, R22, t2], [_0, _0, _0, _1]],
        np.float32,
    )


@numba.njit(error_model='numpy', cache=True)
def inv_intrinsic_matrix(K):
    _0 = np.float32(0)
    _1 = np.float32(1)

    if K[0, 1] == _0:
        L00 = _1 / K[0, 0]
        L11 = _1 / K[1, 1]
        L01 = _0
        L02 = -K[0, 2] * L00
        L12 = -K[1, 2] * L11
    else:
        K12 = K[1, 2]
        L00 = _1 / K[0, 0]
        L11 = _1 / K[1, 1]
        L01 = -K[0, 1] * L00 * L11
        L02 = -L01 * K12 - K[0, 2] * L00
        L12 = -K12 * L11
    return np.array([[L00, L01, L02], [_0, L11, L12], [_0, _0, _1]], np.float32)


@numba.njit(error_model='numpy', cache=True)
def world_to_image(points, K, R, camloc):
    n_points = points.shape[0]
    out = np.empty((n_points, 2), dtype=points.dtype)
    M00, M01, M02, M10, M11, M12, M20, M21, M22 = get_projection_matrix3x3(K, R).flat
    t0, t1, t2 = camloc
    _0 = np.float32(0)
    _1 = np.float32(1)
    _nan = np.float32(np.nan)
    for i in numba.prange(n_points):
        in_x, in_y, in_z = points[i]
        in_x -= t0
        in_y -= t1
        in_z -= t2
        out_z = M20 * in_x + M21 * in_y + M22 * in_z
        if out_z <= _0:
            out[i, 0] = _nan
            out[i, 1] = _nan
        else:
            inv_z = _1 / out_z
            out[i, 0] = (M00 * in_x + M01 * in_y + M02 * in_z) * inv_z
            out[i, 1] = (M10 * in_x + M11 * in_y + M12 * in_z) * inv_z
    return out


@numba.njit(error_model='numpy', cache=True)
def world_to_camera(points, R, camloc):
    n_points = points.shape[0]
    out = np.empty((n_points, 3), dtype=points.dtype)
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = R.flat
    t0, t1, t2 = camloc
    for i in numba.prange(n_points):
        in_x, in_y, in_z = points[i]
        in_x -= t0
        in_y -= t1
        in_z -= t2
        out[i, 0] = R00 * in_x + R01 * in_y + R02 * in_z
        out[i, 1] = R10 * in_x + R11 * in_y + R12 * in_z
        out[i, 2] = R20 * in_x + R21 * in_y + R22 * in_z
    return out


@numba.njit(error_model='numpy', cache=True)
def world_to_undist(points, R, camloc):
    n_points = points.shape[0]
    out = np.empty((n_points, 2), dtype=points.dtype)
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = R.flat
    t0, t1, t2 = camloc
    _0 = np.float32(0)
    _1 = np.float32(1)
    _nan = np.float32(np.nan)
    for i in numba.prange(n_points):
        in_x, in_y, in_z = points[i]
        in_x -= t0
        in_y -= t1
        in_z -= t2
        out_z = R20 * in_x + R21 * in_y + R22 * in_z
        if out_z <= _0:
            out[i, 0] = _nan
            out[i, 1] = _nan
        else:
            inv_z = _1 / out_z
            out[i, 0] = (R00 * in_x + R01 * in_y + R02 * in_z) * inv_z
            out[i, 1] = (R10 * in_x + R11 * in_y + R12 * in_z) * inv_z
    return out


@numba.njit(error_model='numpy', cache=True)
def camera_to_world(points, R, camloc, dst):
    n_points = points.shape[0]
    if dst is None:
        dst = np.empty_like(points)
    R00, R01, R02, R10, R11, R12, R20, R21, R22 = R.flat
    t0, t1, t2 = camloc
    for i in numba.prange(n_points):
        in_x, in_y, in_z = points[i]
        dst[i, 0] = R00 * in_x + R10 * in_y + R20 * in_z + t0
        dst[i, 1] = R01 * in_x + R11 * in_y + R21 * in_z + t1
        dst[i, 2] = R02 * in_x + R12 * in_y + R22 * in_z + t2
    return dst


@numba.njit(error_model='numpy', cache=True)
def mul_K_M_Kinv(K1, M, K2):
    # Computes K1 @ M @ inv(K2)
    return mul_M_K_inv(get_projection_matrix3x3(K1, M), K2)


@numba.njit(error_model='numpy', cache=True)
def mul_M_K_inv(M, K):
    # Computes M @ inv(K)
    L = inv_intrinsic_matrix(K)
    L00 = L[0, 0]
    L01 = L[0, 1]
    L02 = L[0, 2]
    L11 = L[1, 1]
    L12 = L[1, 2]
    M00, M01, M02, M10, M11, M12, M20, M21, M22 = M.flat
    N00 = M00 * L00
    N01 = M00 * L01 + M01 * L11
    N02 = M00 * L02 + M01 * L12 + M02
    N10 = M10 * L00
    N11 = M10 * L01 + M11 * L11
    N12 = M10 * L02 + M11 * L12 + M12
    N20 = M20 * L00
    N21 = M20 * L01 + M21 * L11
    N22 = M20 * L02 + M21 * L12 + M22
    return np.array([[N00, N01, N02], [N10, N11, N12], [N20, N21, N22]], np.float32)


@numba.njit(error_model='numpy', cache=True)
def reproject_intrinsics(points, K_old, K_new):
    out = np.empty_like(points)
    L = relative_intrinsics(K_old, K_new)
    L00 = L[0, 0]
    L01 = L[0, 1]
    L02 = L[0, 2]
    L11 = L[1, 1]
    L12 = L[1, 2]
    for i in range(points.shape[0]):
        p0, p1 = points[i]
        out[i, 0] = p0 * L00 + p1 * L01 + L02
        out[i, 1] = p1 * L11 + L12
    return out


@numba.njit(error_model='numpy', cache=True)
def relative_intrinsics(K_old, K_new):
    _0 = np.float32(0)
    _1 = np.float32(1)

    if K_old[0, 1] == _0 and K_new[0, 1] == _0:
        L00 = K_new[0, 0] / K_old[0, 0]
        L11 = K_new[1, 1] / K_old[1, 1]
        L02 = K_new[0, 2] - L00 * K_old[0, 2]
        L12 = K_new[1, 2] - L11 * K_old[1, 2]
        return np.array([[L00, _0, L02], [_0, L11, L12], [_0, _0, _1]], np.float32)
    else:
        K12o = K_old[1, 2]
        K11o_inv = _1 / K_old[1, 1]
        L00 = K_new[0, 0] / K_old[0, 0]
        L01 = (K_new[0, 1] - L00 * K_old[0, 1]) * K11o_inv
        L02 = K_new[0, 2] - L01 * K12o - L00 * K_old[0, 2]
        L11 = K_new[1, 1] * K11o_inv
        L12 = K_new[1, 2] - L11 * K12o
        return np.array([[L00, L01, L02], [_0, L11, L12], [_0, _0, _1]], np.float32)


@numba.njit(error_model='numpy', cache=True)
def transform_perspective(points, H, dst):
    H00, H01, H02, H10, H11, H12, H20, H21, H22 = H.flat
    _0 = np.float32(0)
    _1 = np.float32(1)
    _nan = np.float32(np.nan)
    if dst is None:
        dst = np.empty_like(points)

    for i in numba.prange(points.shape[0]):
        in_x = points[i, 0]
        in_y = points[i, 1]
        out_z = H20 * in_x + H21 * in_y + H22
        if out_z <= _0:
            dst[i, 0] = _nan
            dst[i, 1] = _nan
        else:
            inv_z = _1 / out_z
            dst[i, 0] = (H00 * in_x + H01 * in_y + H02) * inv_z
            dst[i, 1] = (H10 * in_x + H11 * in_y + H12) * inv_z
    return dst


@numba.njit(error_model='numpy', cache=True)
def reproject_full(points, K_old, R_old, R_new, K_new):
    if np.array_equal(R_old, R_new):
        return reproject_intrinsics(points, K_old, K_new)

    H = mul_K_M_Kinv(K_old, R_old @ R_new.T, K_new)

    points_new = np.empty_like(points)
    n_points = points.shape[0]
    H00, H01, H02, H10, H11, H12, H20, H21, H22 = H.flat
    _0 = np.float32(0)
    _1 = np.float32(1)
    _nan = np.float32(np.nan)
    for i in numba.prange(n_points):
        in_x = points[i, 0]
        in_y = points[i, 1]
        out_z = H20 * in_x + H21 * in_y + H22
        if out_z <= _0:
            points_new[i, 0] = _nan
            points_new[i, 1] = _nan
        else:
            inv_z = _1 / out_z
            points_new[i, 0] = (H00 * in_x + H01 * in_y + H02) * inv_z
            points_new[i, 1] = (H10 * in_x + H11 * in_y + H12) * inv_z
    return points_new


@numba.njit(error_model='numpy', cache=True)
def invert3x3(A):
    A00, A01, A02, A10, A11, A12, A20, A21, A22 = A.flat
    A3746 = A10 * A21 - A11 * A20
    A4857 = A11 * A22 - A12 * A21
    A5638 = A12 * A20 - A10 * A22
    det = A00 * A4857 + A01 * A5638 + A02 * A3746
    inv_det = np.float32(1.0) / det
    return np.array(
        [
            [
                A4857 * inv_det,
                -(A01 * A22 - A02 * A21) * inv_det,
                +(A01 * A12 - A02 * A11) * inv_det,
            ],
            [
                A5638 * inv_det,
                +(A00 * A22 - A02 * A20) * inv_det,
                -(A00 * A12 - A02 * A10) * inv_det,
            ],
            [
                A3746 * inv_det,
                -(A00 * A21 - A01 * A20) * inv_det,
                +(A00 * A11 - A01 * A10) * inv_det,
            ],
        ],
        np.float32,
    )
