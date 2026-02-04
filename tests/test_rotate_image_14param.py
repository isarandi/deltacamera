"""Tests for 14-parameter distortion coefficient rotation.

This implements and tests the rotation of tilt sensor parameters (τx, τy)
which are OpenCV's distortion coefficients 12 and 13.

The tilt model applies a projective transformation after radial/tangential
distortion, modeling a sensor that is not perpendicular to the optical axis.

The key insight for Scheimpflug tilt rotation is that we need to reorder
Euler angles from XYZ to ZXY to push the Z rotation through the tilt.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


# =============================================================================
# Tilt Transformation Functions
# =============================================================================

def tilt_transform(x, y, tau_x, tau_y):
    """Apply tilted sensor transformation to normalized coordinates.

    The tilt is modeled as viewing the image plane through a rotation
    R = Ry(tau_y) @ Rx(tau_x), then reprojecting to z=1.

    Args:
        x, y: Normalized (distorted) coordinates, can be arrays
        tau_x: Tilt angle around x-axis (radians)
        tau_y: Tilt angle around y-axis (radians)

    Returns:
        x_tilt, y_tilt: Coordinates after tilt transformation
    """
    if tau_x == 0 and tau_y == 0:
        return x, y

    cx, sx = np.cos(tau_x), np.sin(tau_x)
    cy, sy = np.cos(tau_y), np.sin(tau_y)

    # R = Ry(tau_y) @ Rx(tau_x) =
    # [[cy,  sx*sy, cx*sy],
    #  [0,   cx,    -sx  ],
    #  [-sy, sx*cy, cx*cy]]

    # Apply homography: project (x, y, 1) through R, then normalize by z
    z = -sy * x + sx * cy * y + cx * cy
    x_tilt = (cy * x + sx * sy * y + cx * sy) / z
    y_tilt = (cx * y - sx) / z

    return x_tilt, y_tilt


def tilt_transform_inverse(x_tilt, y_tilt, tau_x, tau_y):
    """Inverse of tilted sensor transformation.

    Applies R^T (inverse rotation) as a homography.
    """
    if tau_x == 0 and tau_y == 0:
        return x_tilt, y_tilt

    cx, sx = np.cos(tau_x), np.sin(tau_x)
    cy, sy = np.cos(tau_y), np.sin(tau_y)

    # R^T = [[cy,     0,     -sy   ],
    #        [sx*sy,  cx,    sx*cy ],
    #        [cx*sy, -sx,    cx*cy ]]

    z = cx * sy * x_tilt - sx * y_tilt + cx * cy
    x = (cy * x_tilt - sy) / z
    y = (sx * sy * x_tilt + cx * y_tilt + sx * cy) / z

    return x, y


# =============================================================================
# Distortion Functions
# =============================================================================

def distort_brown_conrady_12(x, y, d):
    """Apply 12-parameter Brown-Conrady distortion.

    Args:
        x, y: Normalized undistorted coordinates
        d: [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]
    """
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = d[:12]

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    # Radial distortion (rational model)
    radial_num = 1 + k1 * r2 + k2 * r4 + k3 * r6
    radial_den = 1 + k4 * r2 + k5 * r4 + k6 * r6
    radial = radial_num / radial_den

    # Tangential distortion
    xy2 = 2 * x * y
    tang_x = p1 * xy2 + p2 * (r2 + 2 * x * x)
    tang_y = p1 * (r2 + 2 * y * y) + p2 * xy2

    # Thin prism distortion
    prism_x = s1 * r2 + s2 * r4
    prism_y = s3 * r2 + s4 * r4

    x_dist = x * radial + tang_x + prism_x
    y_dist = y * radial + tang_y + prism_y

    return x_dist, y_dist


def distort_14param(x, y, d):
    """Apply 14-parameter distortion (12-param Brown-Conrady + tilt).

    Args:
        x, y: Normalized undistorted coordinates
        d: 14-element array [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, τx, τy]
    """
    # First apply 12-param distortion
    x_dist, y_dist = distort_brown_conrady_12(x, y, d[:12])

    # Then apply tilt transformation
    tau_x, tau_y = d[12], d[13]
    x_tilt, y_tilt = tilt_transform(x_dist, y_dist, tau_x, tau_y)

    return x_tilt, y_tilt


# =============================================================================
# Projection Functions
# =============================================================================

def project_points_14param(points_3d, K, d, E=None):
    """Project 3D points to image using 14-param distortion.

    Args:
        points_3d: (N, 3) array of points in world coordinates
        K: 3x3 intrinsic matrix
        d: 14-element distortion coefficients
        E: 3x3 extrinsic rotation matrix, default identity

    Returns:
        (N, 2) array of image points
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    # Apply extrinsic rotation if provided
    if E is not None:
        points_3d = (E @ points_3d.T).T

    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    # Normalize
    x = X / Z
    y = Y / Z

    # Apply 14-param distortion
    x_dist, y_dist = distort_14param(x, y, d)

    # Apply intrinsic matrix (including possible skew)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    skew = K[0, 1]

    u = fx * x_dist + skew * y_dist + cx
    v = fy * y_dist + cy

    return np.stack([u, v], axis=1)


# =============================================================================
# Rotation Functions (Exact Implementation)
# =============================================================================

def rotate_camera_params_14param(E, K, d, theta, o):
    """Rotate camera parameters for image rotation by theta around anchor o.

    This is the exact transformation that satisfies:
        project(p; E', K', d') = R_theta @ (project(p; E, K, d) - o) + o

    For the Scheimpflug (tilt) case, we need to:
    1. Reorder Euler angles XYZ -> ZXY to push Z rotation through tilt
    2. Apply additional K correction for the tilt homography change

    Args:
        E: 3x3 extrinsic rotation matrix
        K: 3x3 intrinsic matrix
        d: 14-element distortion array
        theta: Rotation angle (radians, CCW positive)
        o: 2D anchor point for rotation

    Returns:
        E_new, K_new, d_new: Transformed camera parameters
    """
    E = np.array(E, dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    d = np.array(d, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)

    # 2D rotation matrix
    R = Rotation.from_euler('z', theta).as_matrix()[:2, :2]

    # Build R_ correction (ensures K[1,0] = 0, no vertical skew)
    x = R[1, :] @ K[:2, :2]
    x = x / np.linalg.norm(x)
    R_ = np.array([[x[1], -x[0]], [x[0], x[1]]])

    # Transform intrinsic matrix (before tilt correction)
    K_new = K.copy()
    K_new[:2, :2] = R @ K[:2, :2] @ R_.T
    K_new[:2, 2] = R @ (K[:2, 2] - o) + o

    # Transform distortion coefficients
    d_new = d.copy()

    # Check if we have tilt
    has_tilt = len(d) >= 14 and (d[12] != 0 or d[13] != 0)

    if has_tilt:
        # Scheimpflug tilt case: need Euler angle reordering

        # Build the tilt rotation matrix
        sch = Rotation.from_euler('xy', [d[12], d[13]]).as_matrix()

        # The tilt produces a homography B that we need to track
        sch_B = np.array([
            [sch[2, 2], 0, -sch[0, 2]],
            [0, sch[2, 2], -sch[1, 2]],
            [0, 0, 1]])

        # R_ as 3x3
        R_3x3 = np.array([
            [R_[0, 0], R_[0, 1], 0],
            [R_[1, 0], R_[1, 1], 0],
            [0, 0, 1]])

        # Transformed homography
        sch_A = R_3x3 @ sch_B @ R_3x3.T

        # Reorder Euler angles: XYZ -> ZXY
        # This gives us the Z rotation that appears AFTER tilt in the new decomposition
        theta_ = np.arctan2(x[0], x[1])
        theta__, d_new[12], d_new[13] = Rotation.from_euler(
            'xyz', [d[12], d[13], theta_]).as_euler('zxy')

        # R__ is the rotation to use for tangential/thin-prism and extrinsics
        R__ = Rotation.from_euler('z', theta__).as_matrix()[:2, :2]

        # Build inverse of new tilt homography
        sch_new = Rotation.from_euler('xy', [d_new[12], d_new[13]]).as_matrix()
        sch_new_r22 = 1 / sch_new[2, 2]
        sch_B_new_inv = np.array([
            [sch_new_r22, 0, sch_new[0, 2] * sch_new_r22],
            [0, sch_new_r22, sch_new[1, 2] * sch_new_r22],
            [0, 0, 1]])

        # Apply tilt homography correction to K
        K_new = K_new @ sch_A @ sch_B_new_inv
    else:
        # No tilt: use R_ for tangential/thin-prism rotation
        R__ = R_

    # Rotate tangential (p1, p2) and thin prism (s1, s2, s3, s4)
    # d[[3, 8, 9]] = [p2, s1, s2]
    # d[[2, 10, 11]] = [p1, s3, s4]
    tang_prism = d_new[[[3, 8, 9], [2, 10, 11]]]
    d_new[[[3, 8, 9], [2, 10, 11]]] = R__ @ tang_prism

    # Transform extrinsic rotation matrix
    E_new = E.copy()
    E_new[:2] = R__ @ E[:2]

    return E_new, K_new, d_new


# =============================================================================
# Horizontal Flip Functions
# =============================================================================

def flip_camera_params_14param(E, K, d, imshape):
    """Flip camera parameters for horizontal image flip.

    This transforms the camera so that projecting gives the same result
    as projecting with original params and then flipping the image.

    Args:
        E: 3x3 extrinsic rotation matrix
        K: 3x3 intrinsic matrix
        d: 14-element distortion array
        imshape: (height, width)

    Returns:
        E_new, K_new, d_new: Transformed camera parameters
    """
    E = np.array(E, dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    d = np.array(d, dtype=np.float64)

    # Flip extrinsics: negate first row (x-axis)
    E_new = E.copy()
    E_new[0] *= -1

    # Flip intrinsics: mirror principal point and negate skew
    K_new = K.copy()
    K_new[0, 2] = (imshape[1] - 1) - K[0, 2]
    K_new[0, 1] *= -1  # Negate skew

    # Flip distortion coefficients
    d_new = d.copy()
    # p2 at index 3
    d_new[3] *= -1
    # s1, s2 at indices 8, 9
    d_new[8] *= -1
    d_new[9] *= -1
    # τy at index 13 (if present)
    if len(d) >= 14:
        d_new[13] *= -1

    return E_new, K_new, d_new


# =============================================================================
# Helper: Manual Pixel Transformations
# =============================================================================

def flip_pixels(points_2d, imshape):
    """Flip 2D pixel coordinates horizontally."""
    points = np.asarray(points_2d).copy()
    points[:, 0] = (imshape[1] - 1) - points[:, 0]
    return points


def rotate_pixels(points_2d, theta, imshape, anchor=None):
    """Rotate 2D pixel coordinates by theta around anchor."""
    if anchor is None:
        h, w = imshape[:2]
        anchor = np.array([(w - 1) / 2, (h - 1) / 2])
    else:
        anchor = np.asarray(anchor)

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    points = np.asarray(points_2d)
    centered = points - anchor
    rotated = (R @ centered.T).T

    return rotated + anchor


# =============================================================================
# Tests
# =============================================================================

class TestTiltTransform:
    """Test the tilt transformation itself."""

    def test_identity_tilt(self):
        """Zero tilt should be identity."""
        x, y = 0.5, -0.3
        x_t, y_t = tilt_transform(x, y, 0, 0)
        np.testing.assert_allclose([x_t, y_t], [x, y])

    def test_tilt_inverse_roundtrip(self):
        """Tilt followed by inverse should return original."""
        tau_x, tau_y = 0.05, -0.08

        x = np.linspace(-0.3, 0.3, 10)
        y = np.linspace(-0.2, 0.2, 10)

        x_t, y_t = tilt_transform(x, y, tau_x, tau_y)
        x_r, y_r = tilt_transform_inverse(x_t, y_t, tau_x, tau_y)

        np.testing.assert_allclose(x_r, x, rtol=1e-10)
        np.testing.assert_allclose(y_r, y, rtol=1e-10)

    def test_tilt_affects_points(self):
        """Non-zero tilt should change coordinates."""
        tau_x, tau_y = 0.1, 0.05
        x, y = 0.2, 0.15

        x_t, y_t = tilt_transform(x, y, tau_x, tau_y)

        assert not np.isclose(x_t, x) or not np.isclose(y_t, y)


class TestRotateImage14Param:
    """Test that rotate_image works correctly with 14-param distortion.

    The core test: project points, rotate pixels manually, vs
    rotate camera parameters and then project. Results should match exactly
    (up to floating point precision).
    """

    @pytest.mark.parametrize("theta", [np.pi/6, np.pi/2, np.pi, -np.pi/4, 0.73])
    def test_rotation_consistency(self, theta):
        """Project→rotate pixels should match rotate_params→project."""
        # 14-param distortion with awkward values
        d = np.array([
            -0.10457,   # k1
            0.03821,    # k2
            0.00173,    # p1
            -0.00289,   # p2
            0.01534,    # k3
            0.00823,    # k4
            -0.00412,   # k5
            0.00156,    # k6
            0.00067,    # s1
            -0.00089,   # s2
            0.00034,    # s3
            -0.00045,   # s4
            0.04123,    # τx (tilt around x)
            -0.03567,   # τy (tilt around y)
        ], dtype=np.float64)

        K = np.array([
            [517.3, 0, 327.8],
            [0, 518.9, 241.2],
            [0, 0, 1]
        ], dtype=np.float64)

        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)
        o = np.array([(imshape[1] - 1) / 2, (imshape[0] - 1) / 2])

        # Generate test points
        np.random.seed(42)
        world_points = np.random.randn(50, 3).astype(np.float64) * 0.08
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2.5

        # Method 1: Project with original params, then rotate pixels
        img_orig = project_points_14param(world_points, K, d, E)
        img_rotated_manual = rotate_pixels(img_orig, theta, imshape)

        # Method 2: Rotate all params (E, K, d), then project
        E_rot, K_rot, d_rot = rotate_camera_params_14param(E, K, d, theta, o)
        img_rotated_proj = project_points_14param(world_points, K_rot, d_rot, E_rot)

        # Compare - should be exact up to floating point
        np.testing.assert_allclose(
            img_rotated_proj,
            img_rotated_manual,
            atol=1e-9,
            err_msg=f"Rotation by {np.degrees(theta):.1f}° failed"
        )

    def test_tilt_only(self):
        """Test rotation with only tilt distortion (no radial/tangential)."""
        d = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,  # No radial/tangential
            0, 0, 0, 0,              # No thin prism
            0.08, -0.06,             # Only tilt
        ], dtype=np.float64)

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)
        theta = np.pi / 3
        o = np.array([(imshape[1] - 1) / 2, (imshape[0] - 1) / 2])

        np.random.seed(456)
        world_points = np.random.randn(40, 3).astype(np.float64) * 0.15
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        img_orig = project_points_14param(world_points, K, d, E)
        img_rotated_manual = rotate_pixels(img_orig, theta, imshape)

        E_rot, K_rot, d_rot = rotate_camera_params_14param(E, K, d, theta, o)
        img_rotated_proj = project_points_14param(world_points, K_rot, d_rot, E_rot)

        np.testing.assert_allclose(img_rotated_proj, img_rotated_manual, atol=1e-9)

    def test_no_tilt(self):
        """Test rotation with 12-param distortion (no tilt)."""
        d = np.array([
            -0.10457, 0.03821, 0.00173, -0.00289, 0.01534,
            0.00823, -0.00412, 0.00156,
            0.00067, -0.00089, 0.00034, -0.00045,
            0, 0,  # No tilt
        ], dtype=np.float64)

        K = np.array([[517.3, 0, 327.8], [0, 518.9, 241.2], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)
        theta = 0.73
        o = np.array([(imshape[1] - 1) / 2, (imshape[0] - 1) / 2])

        np.random.seed(123)
        world_points = np.random.randn(30, 3).astype(np.float64) * 0.1
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        img_orig = project_points_14param(world_points, K, d, E)
        img_rotated_manual = rotate_pixels(img_orig, theta, imshape)

        E_rot, K_rot, d_rot = rotate_camera_params_14param(E, K, d, theta, o)
        img_rotated_proj = project_points_14param(world_points, K_rot, d_rot, E_rot)

        np.testing.assert_allclose(img_rotated_proj, img_rotated_manual, atol=1e-9)

    def test_roundtrip_rotation(self):
        """Rotate by θ then -θ should return original projections."""
        d = np.array([
            -0.09123, 0.03456, 0.00189, -0.00234, 0.01234,
            0.00678, -0.00345, 0.00167,
            0.00078, -0.00091, 0.00045, -0.00052,
            0.06789, -0.05432,
        ], dtype=np.float64)

        K = np.array([[510, 0, 325], [0, 510, 238], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)
        theta = 0.73
        o = np.array([(imshape[1] - 1) / 2, (imshape[0] - 1) / 2])

        np.random.seed(789)
        world_points = np.random.randn(35, 3).astype(np.float64) * 0.1
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        img_orig = project_points_14param(world_points, K, d, E)

        # Rotate by theta
        E1, K1, d1 = rotate_camera_params_14param(E, K, d, theta, o)

        # Rotate by -theta
        E2, K2, d2 = rotate_camera_params_14param(E1, K1, d1, -theta, o)

        img_roundtrip = project_points_14param(world_points, K2, d2, E2)

        np.testing.assert_allclose(img_roundtrip, img_orig, atol=1e-9)

    def test_nonsquare_pixels(self):
        """Test with non-square pixels (fx ≠ fy) - should still be exact."""
        d = np.array([
            -0.11234, 0.04567, 0.00212, -0.00345, 0.01678,
            0.00789, -0.00456, 0.00234,
            0.00089, -0.00123, 0.00056, -0.00067,
            0.03456, -0.02789,
        ], dtype=np.float64)

        # Non-square pixels
        K = np.array([[520, 0, 330], [0, 480, 245], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)
        theta = np.pi / 4
        o = np.array([(imshape[1] - 1) / 2, (imshape[0] - 1) / 2])

        np.random.seed(321)
        world_points = np.random.randn(40, 3).astype(np.float64) * 0.1
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2.5

        img_orig = project_points_14param(world_points, K, d, E)
        img_rotated_manual = rotate_pixels(img_orig, theta, imshape)

        E_rot, K_rot, d_rot = rotate_camera_params_14param(E, K, d, theta, o)
        img_rotated_proj = project_points_14param(world_points, K_rot, d_rot, E_rot)

        # Should be exact even with non-square pixels
        np.testing.assert_allclose(img_rotated_proj, img_rotated_manual, atol=1e-9)

    def test_90deg_rotation(self):
        """Test exact 90° rotations."""
        d = np.array([
            -0.08765, 0.02345, 0.00134, -0.00267, 0.00912,
            0.00567, -0.00234, 0.00123,
            0.00045, -0.00056, 0.00023, -0.00034,
            0.05678, -0.04321,
        ], dtype=np.float64)

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)
        o = np.array([(imshape[1] - 1) / 2, (imshape[0] - 1) / 2])

        np.random.seed(123)
        world_points = np.random.randn(30, 3).astype(np.float64) * 0.1
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        for k in [1, 2, 3]:
            theta = k * np.pi / 2
            img_orig = project_points_14param(world_points, K, d, E)
            img_rotated_manual = rotate_pixels(img_orig, theta, imshape)

            E_rot, K_rot, d_rot = rotate_camera_params_14param(E, K, d, theta, o)
            img_rotated_proj = project_points_14param(world_points, K_rot, d_rot, E_rot)

            np.testing.assert_allclose(
                img_rotated_proj, img_rotated_manual, atol=1e-9,
                err_msg=f"90° rotation k={k} failed"
            )


class TestHorizontalFlip14Param:
    """Test horizontal_flip_image for 14-param distortion."""

    def test_flip_consistency(self):
        """Project→flip pixels should match flip_params→project.

        Uses non-square pixels and skew by default for thorough testing.
        """
        d = np.array([
            -0.10457, 0.03821, 0.00173, -0.00289, 0.01534,
            0.00823, -0.00412, 0.00156,
            0.00067, -0.00089, 0.00034, -0.00045,
            0.04123, -0.03567,
        ], dtype=np.float64)

        # Non-square pixels with skew
        K = np.array([[517.3, 2.5, 327.8], [0, 489.1, 241.2], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)

        np.random.seed(42)
        world_points = np.random.randn(50, 3).astype(np.float64) * 0.08
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2.5

        img_orig = project_points_14param(world_points, K, d, E)
        img_flipped_manual = flip_pixels(img_orig, imshape)

        E_flip, K_flip, d_flip = flip_camera_params_14param(E, K, d, imshape)
        img_flipped_proj = project_points_14param(world_points, K_flip, d_flip, E_flip)

        np.testing.assert_allclose(img_flipped_proj, img_flipped_manual, atol=1e-9)

    def test_flip_tilt_only(self):
        """Test flip with only tilt distortion, non-square pixels and skew."""
        d = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0,
            0.06789, -0.05432,
        ], dtype=np.float64)

        # Non-square pixels with skew
        K = np.array([[520, 1.8, 320], [0, 475, 240], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)

        np.random.seed(456)
        world_points = np.random.randn(40, 3).astype(np.float64) * 0.15
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        img_orig = project_points_14param(world_points, K, d, E)
        img_flipped_manual = flip_pixels(img_orig, imshape)

        E_flip, K_flip, d_flip = flip_camera_params_14param(E, K, d, imshape)
        img_flipped_proj = project_points_14param(world_points, K_flip, d_flip, E_flip)

        np.testing.assert_allclose(img_flipped_proj, img_flipped_manual, atol=1e-9)

    def test_flip_roundtrip(self):
        """Flip twice should return original projections."""
        d = np.array([
            -0.09123, 0.03456, 0.00189, -0.00234, 0.01234,
            0.00678, -0.00345, 0.00167,
            0.00078, -0.00091, 0.00045, -0.00052,
            0.06789, -0.05432,
        ], dtype=np.float64)

        # Non-square pixels with skew
        K = np.array([[510, 3.2, 325], [0, 485, 238], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)

        np.random.seed(789)
        world_points = np.random.randn(35, 3).astype(np.float64) * 0.1
        world_points[:, 2] = np.abs(world_points[:, 2]) + 2

        img_orig = project_points_14param(world_points, K, d, E)

        E1, K1, d1 = flip_camera_params_14param(E, K, d, imshape)
        E2, K2, d2 = flip_camera_params_14param(E1, K1, d1, imshape)

        img_roundtrip = project_points_14param(world_points, K2, d2, E2)

        np.testing.assert_allclose(img_roundtrip, img_orig, atol=1e-10)

    def test_flip_large_tilt_nonsquare_skew(self):
        """Test flip with large tilt, non-square pixels and skew."""
        d = np.array([
            -0.08765, 0.02345, 0.00134, -0.00267, 0.00912,
            0.00567, -0.00234, 0.00123,
            0.00045, -0.00056, 0.00023, -0.00034,
            0.45, -0.38,  # Large tilt
        ], dtype=np.float64)

        # Non-square pixels with significant skew
        K = np.array([[530, 4.5, 315], [0, 470, 245], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)

        np.random.seed(111)
        world_points = np.random.randn(40, 3).astype(np.float64) * 0.05
        world_points[:, 2] = np.abs(world_points[:, 2]) + 3

        img_orig = project_points_14param(world_points, K, d, E)
        img_flipped_manual = flip_pixels(img_orig, imshape)

        E_flip, K_flip, d_flip = flip_camera_params_14param(E, K, d, imshape)
        img_flipped_proj = project_points_14param(world_points, K_flip, d_flip, E_flip)

        np.testing.assert_allclose(img_flipped_proj, img_flipped_manual, atol=1e-9)


class TestTiltFlip:
    """Test tilt flip transformation directly."""

    def test_tilt_flip_derivation(self):
        """Verify the flip derivation: tilt(-x, y, τx, -τy) = (-x_t, y_t)."""
        tau_x, tau_y = 0.05, -0.08

        x = np.linspace(-0.3, 0.3, 20)
        y = np.linspace(-0.2, 0.2, 20)

        x_t, y_t = tilt_transform(x, y, tau_x, tau_y)
        x_t_flip, y_t_flip = tilt_transform(-x, y, tau_x, -tau_y)

        np.testing.assert_allclose(x_t_flip, -x_t, rtol=1e-10)
        np.testing.assert_allclose(y_t_flip, y_t, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_rotation(self):
        """Zero rotation should not change parameters."""
        d = np.array([
            -0.1, 0.05, 0.001, -0.002, 0.01,
            0.005, -0.003, 0.002,
            0.0005, -0.0006, 0.0003, -0.0004,
            0.05, -0.04,
        ], dtype=np.float64)

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)
        o = np.array([(imshape[1] - 1) / 2, (imshape[0] - 1) / 2])

        E_rot, K_rot, d_rot = rotate_camera_params_14param(E, K, d, 0, o)

        np.testing.assert_allclose(d_rot, d, atol=1e-15)
        np.testing.assert_allclose(K_rot, K, atol=1e-15)
        np.testing.assert_allclose(E_rot, E, atol=1e-15)

    def test_large_tilt(self):
        """Test with large tilt values."""
        d = np.array([
            -0.1, 0.05, 0.001, -0.002, 0.01,
            0.005, -0.003, 0.002,
            0.0005, -0.0006, 0.0003, -0.0004,
            0.5, -0.4,  # Large tilt ~30 degrees
        ], dtype=np.float64)

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        E = np.eye(3, dtype=np.float64)
        imshape = (480, 640)
        o = np.array([(imshape[1] - 1) / 2, (imshape[0] - 1) / 2])
        theta = np.pi / 3

        np.random.seed(999)
        world_points = np.random.randn(30, 3).astype(np.float64) * 0.05
        world_points[:, 2] = np.abs(world_points[:, 2]) + 3

        img_orig = project_points_14param(world_points, K, d, E)
        img_rotated_manual = rotate_pixels(img_orig, theta, imshape)

        E_rot, K_rot, d_rot = rotate_camera_params_14param(E, K, d, theta, o)
        img_rotated_proj = project_points_14param(world_points, K_rot, d_rot, E_rot)

        np.testing.assert_allclose(img_rotated_proj, img_rotated_manual, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
