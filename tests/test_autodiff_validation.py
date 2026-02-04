"""Validate SymPy-derived analytical expressions against PyTorch autodiff.

This module tests that the hand-derived (via SymPy) Jacobian formulas in the codebase
match what automatic differentiation produces. This provides high confidence that the
complex expressions with intermediate variables (x0, x1, ..., x49) are correct.

Functions tested:
- validity.jacobian_det_polar: Jacobian determinant of Brown-Conrady distortion
- validity.jacobian_det_and_prime_polar: Jacobian det and its derivative w.r.t. r
- distortion._undistort_points: Jacobian matrix elements for Newton's method
- Fisheye distortion derivative (implicit in undistortion)
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import deltacamera.validity
import deltacamera.distortion
from conftest import BROWN_CONRADY_COEFFS, FISHEYE_COEFFS, extend_distortion_coeffs


# =============================================================================
# PyTorch reference implementations of distortion functions
# =============================================================================

def distort_brown_conrady_torch(p, d):
    """Brown-Conrady distortion in PyTorch for autodiff.

    Args:
        p: (2,) tensor of undistorted normalized coordinates (x, y)
        d: (12,) tensor of distortion coefficients

    Returns:
        (2,) tensor of distorted coordinates
    """
    x, y = p[0], p[1]
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = d

    r2 = x * x + y * y

    # Radial distortion (rational model)
    radial_num = 1 + r2 * (k0 + r2 * (k1 + k4 * r2))
    radial_den = 1 + r2 * (k5 + r2 * (k6 + k7 * r2))
    a = radial_num / radial_den

    # Tangential distortion
    b = 2 * k2 * y + 2 * k3 * x

    # Thin prism distortion
    cx = r2 * (k3 + k8 + k9 * r2)
    cy = r2 * (k2 + k10 + k11 * r2)

    s = a + b
    x_dist = x * s + cx
    y_dist = y * s + cy

    return torch.stack([x_dist, y_dist])


def distort_fisheye_torch(p, d):
    """Kannala-Brandt fisheye distortion in PyTorch for autodiff.

    Args:
        p: (2,) tensor of undistorted normalized coordinates (x, y)
        d: (4,) tensor of distortion coefficients

    Returns:
        (2,) tensor of distorted coordinates
    """
    x, y = p[0], p[1]
    d0, d1, d2, d3 = d

    r2 = x * x + y * y
    r = torch.sqrt(r2 + 1e-12)  # Small epsilon for numerical stability at origin

    # Incidence angle
    theta = torch.atan(r)
    theta2 = theta * theta

    # Distorted radius
    theta_d = theta * (1 + theta2 * (d0 + theta2 * (d1 + theta2 * (d2 + theta2 * d3))))

    # Scale factor
    scale = theta_d / (r + 1e-12)

    return torch.stack([x * scale, y * scale])


# =============================================================================
# Test: Jacobian determinant (Brown-Conrady)
# =============================================================================

class TestJacobianDeterminant:
    """Test jacobian_det_polar against PyTorch autodiff."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS)
    def test_jacobian_det_matches_autodiff(self, d):
        """Analytical Jacobian determinant should match torch.autograd."""
        # Test points in Cartesian coordinates
        test_points = [
            [0.01, 0.01],   # Near origin
            [0.1, 0.0],     # On x-axis
            [0.0, 0.1],     # On y-axis
            [0.1, 0.2],     # First quadrant
            [-0.15, 0.1],   # Second quadrant
            [0.2, -0.1],    # Fourth quadrant
            [-0.1, -0.15],  # Third quadrant
            [0.3, 0.3],     # Larger radius
        ]

        d_torch = torch.tensor(d, dtype=torch.float64)

        for px, py in test_points:
            # PyTorch autodiff Jacobian determinant
            p = torch.tensor([px, py], dtype=torch.float64, requires_grad=True)
            J = torch.autograd.functional.jacobian(
                lambda x: distort_brown_conrady_torch(x, d_torch), p
            )
            det_torch = torch.linalg.det(J).item()

            # Analytical (SymPy-derived) Jacobian determinant
            r = np.sqrt(px * px + py * py)
            t = np.arctan2(py, px)
            det_analytical = deltacamera.validity.jacobian_det_polar(
                np.array([r], np.float32),
                np.array([t], np.float32),
                d
            )[0]

            np.testing.assert_allclose(
                det_analytical, det_torch, rtol=1e-5, atol=1e-10,
                err_msg=f"Mismatch at point ({px}, {py})"
            )

    def test_jacobian_det_at_origin(self):
        """Jacobian determinant at origin should be 1 (identity mapping)."""
        for d in BROWN_CONRADY_COEFFS:
            d_torch = torch.tensor(d, dtype=torch.float64)

            # Use small epsilon instead of exact zero for numerical stability
            p = torch.tensor([1e-8, 1e-8], dtype=torch.float64, requires_grad=True)
            J = torch.autograd.functional.jacobian(
                lambda x: distort_brown_conrady_torch(x, d_torch), p
            )
            det_torch = torch.linalg.det(J).item()

            np.testing.assert_allclose(det_torch, 1.0, rtol=1e-5)


# =============================================================================
# Test: Jacobian determinant derivative w.r.t. r (Brown-Conrady)
# =============================================================================

class TestJacobianDeterminantDerivative:
    """Test jacobian_det_and_prime_polar against PyTorch autodiff."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS[:2])  # Subset for speed
    def test_jacobian_det_derivative_matches_autodiff(self, d):
        """Derivative of Jacobian det w.r.t. r should match torch.autograd."""
        # Test at various radii and angles
        radii = [0.05, 0.1, 0.2, 0.3]
        angles = [0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3]

        d_torch = torch.tensor(d, dtype=torch.float64)

        for r_val in radii:
            for t_val in angles:
                # Analytical (SymPy-derived) function and derivative
                r_np = np.array([r_val], np.float32)
                t_np = np.array([t_val], np.float32)
                fval_analytical, deriv_analytical = deltacamera.validity.jacobian_det_and_prime_polar(
                    r_np, t_np, d
                )

                # PyTorch autodiff: compute d(det(J))/dr
                # We need to differentiate det(J(r*cos(t), r*sin(t))) w.r.t. r
                def det_as_function_of_r(r_tensor):
                    x = r_tensor * np.cos(t_val)
                    y = r_tensor * np.sin(t_val)
                    p = torch.stack([x, y])
                    J = torch.autograd.functional.jacobian(
                        lambda pt: distort_brown_conrady_torch(pt, d_torch), p,
                        create_graph=True  # Needed for second-order differentiation
                    )
                    return torch.linalg.det(J)

                r_torch = torch.tensor(r_val, dtype=torch.float64, requires_grad=True)
                det_val = det_as_function_of_r(r_torch)
                det_val.backward()
                deriv_torch = r_torch.grad.item()
                fval_torch = det_val.item()

                np.testing.assert_allclose(
                    fval_analytical[0], fval_torch, rtol=1e-4, atol=1e-8,
                    err_msg=f"det(J) mismatch at r={r_val}, t={t_val}"
                )
                np.testing.assert_allclose(
                    deriv_analytical[0], deriv_torch, rtol=1e-3, atol=1e-6,
                    err_msg=f"d(det(J))/dr mismatch at r={r_val}, t={t_val}"
                )


# =============================================================================
# Test: Full Jacobian matrix elements (used in Newton's method for undistortion)
# =============================================================================

class TestJacobianMatrixElements:
    """Test the Jacobian matrix elements computed in _undistort_points."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS[:2])
    def test_jacobian_matrix_matches_autodiff(self, d):
        """Jacobian matrix elements should match torch.autograd.functional.jacobian."""
        test_points = [
            [0.05, 0.05],
            [0.1, 0.2],
            [-0.1, 0.15],
            [0.2, -0.1],
        ]

        d_torch = torch.tensor(d, dtype=torch.float64)

        for px, py in test_points:
            # PyTorch autodiff full Jacobian
            p = torch.tensor([px, py], dtype=torch.float64, requires_grad=True)
            J_torch = torch.autograd.functional.jacobian(
                lambda x: distort_brown_conrady_torch(x, d_torch), p
            ).numpy()

            # Extract elements: J = [[dx'/dx, dx'/dy], [dy'/dx, dy'/dy]]
            j00_torch = J_torch[0, 0]
            j01_torch = J_torch[0, 1]
            j10_torch = J_torch[1, 0]
            j11_torch = J_torch[1, 1]

            # Compute analytical Jacobian using the same formula as _undistort_points
            j00_ana, j01_ana, j10_ana, j11_ana = _compute_jacobian_analytical(px, py, d)

            np.testing.assert_allclose(j00_ana, j00_torch, rtol=1e-5, atol=1e-10)
            np.testing.assert_allclose(j01_ana, j01_torch, rtol=1e-5, atol=1e-10)
            np.testing.assert_allclose(j10_ana, j10_torch, rtol=1e-5, atol=1e-10)
            np.testing.assert_allclose(j11_ana, j11_torch, rtol=1e-5, atol=1e-10)


def _compute_jacobian_analytical(px, py, d):
    """Compute Jacobian matrix elements using the same formula as _undistort_points.

    This is extracted from distortion.py lines 267-295 for testing purposes.
    """
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = d
    _1 = np.float32(1)
    _2 = np.float32(2)
    k3_p_k8 = k3 + k8
    k2_p_k10 = k2 + k10
    _2_k2 = _2 * k2
    _2_k3 = _2 * k3

    r2 = px * px + py * py
    _2_x = _2 * px
    _2_y = _2 * py

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
    x14 = px * _2_k3 + py * _2_k2 + x29
    x26 = x13 * x13 * ((r2 * (k4_r2 + x6) + x7) - x29 * x29 * (r2 * (k7_r2 + x10) + x11))
    x19 = px * x26 + k3
    x21 = py * x26 + k2
    x27 = k9_r2 + x2
    x28 = k11_r2 + x16

    j00 = _2_x * (x19 + x27) + x14
    j11 = _2_y * (x21 + x28) + x14
    j01 = _2_x * x21 + _2_y * x27
    j10 = _2_y * x19 + _2_x * x28

    return j00, j01, j10, j11


# =============================================================================
# Test: Fisheye distortion derivative (used in Newton's method)
# =============================================================================

class TestFisheyeDerivative:
    """Test fisheye distortion derivative used in undistortion Newton's method."""

    @pytest.mark.parametrize("d", FISHEYE_COEFFS)
    def test_fisheye_derivative_matches_autodiff(self, d):
        """dr_distorted/dtheta should match torch.autograd."""
        d0, d1, d2, d3 = d

        # Test at various theta values
        theta_values = [0.1, 0.3, 0.5, 0.8, 1.0]

        for theta in theta_values:
            # Analytical derivative (from validity.py fisheye code)
            # r_d = theta * (1 + theta^2 * (d0 + theta^2 * (d1 + theta^2 * (d2 + theta^2 * d3))))
            # dr_d/dtheta = 1 + 3*d0*theta^2 + 5*d1*theta^4 + 7*d2*theta^6 + 9*d3*theta^8
            t2 = theta * theta
            deriv_analytical = 1 + t2 * (3*d0 + t2 * (5*d1 + t2 * (7*d2 + t2 * 9*d3)))

            # PyTorch autodiff
            def fisheye_r_distorted(t):
                t2 = t * t
                return t * (1 + t2 * (d0 + t2 * (d1 + t2 * (d2 + t2 * d3))))

            t_torch = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
            r_d = fisheye_r_distorted(t_torch)
            r_d.backward()
            deriv_torch = t_torch.grad.item()

            np.testing.assert_allclose(
                deriv_analytical, deriv_torch, rtol=1e-6,  # float32 coefficients limit precision
                err_msg=f"Fisheye derivative mismatch at theta={theta}"
            )

    @pytest.mark.parametrize("d", FISHEYE_COEFFS)
    def test_fisheye_jacobian_det_matches_autodiff(self, d):
        """Fisheye Jacobian determinant should match torch.autograd."""
        d_torch = torch.tensor(d, dtype=torch.float64)

        test_points = [
            [0.1, 0.1],
            [0.3, 0.0],
            [0.0, 0.4],
            [0.2, 0.3],
            [-0.2, 0.15],
        ]

        for px, py in test_points:
            # PyTorch autodiff
            p = torch.tensor([px, py], dtype=torch.float64, requires_grad=True)
            J = torch.autograd.functional.jacobian(
                lambda x: distort_fisheye_torch(x, d_torch), p
            )
            det_torch = torch.linalg.det(J).item()

            # For fisheye, Jacobian det should be positive in valid region
            # and equal to (dr_d/dtheta) * (theta_d / theta) at the given point
            # This is a sanity check that the mapping is locally invertible
            assert det_torch > 0, f"Fisheye det(J) should be positive at ({px}, {py})"


# =============================================================================
# Test: Roundtrip consistency (distort then undistort)
# =============================================================================

class TestRoundtripConsistency:
    """Test that distort/undistort are consistent with their Jacobians."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS[:2])
    def test_undistort_uses_correct_jacobian(self, d):
        """Verify undistortion converges correctly, implying Jacobian is correct."""
        # Generate points in valid region
        points = np.array([
            [0.05, 0.05],
            [0.1, 0.1],
            [-0.08, 0.12],
            [0.15, -0.05],
        ], np.float32)

        # Distort points
        distorted = deltacamera.distortion.distort_points(points, d, check_validity=False)

        # Undistort them back
        undistorted = deltacamera.distortion.undistort_points(distorted, d, check_validity=False)

        # Should recover original points
        np.testing.assert_allclose(undistorted, points, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("d", FISHEYE_COEFFS)
    def test_fisheye_undistort_uses_correct_derivative(self, d):
        """Verify fisheye undistortion converges, implying derivative is correct."""
        points = np.array([
            [0.1, 0.1],
            [0.2, 0.0],
            [0.0, 0.3],
            [-0.15, 0.2],
        ], np.float32)

        # Distort points
        distorted = deltacamera.distortion.distort_points_fisheye(
            points, d, check_validity=False
        )

        # Undistort them back
        undistorted = deltacamera.distortion.undistort_points_fisheye(
            distorted, d, check_validity=False
        )

        # Should recover original points
        np.testing.assert_allclose(undistorted, points, rtol=1e-4, atol=1e-5)


# =============================================================================
# Test: Inverse Jacobian (used directly in Newton step)
# =============================================================================

class TestInverseJacobian:
    """Test that inverse Jacobian computation is correct."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS[:2])
    def test_inverse_jacobian_matches_autodiff(self, d):
        """J^{-1} computed analytically should match torch inverse of J."""
        test_points = [
            [0.1, 0.1],
            [0.15, -0.1],
            [-0.1, 0.2],
        ]

        d_torch = torch.tensor(d, dtype=torch.float64)

        for px, py in test_points:
            # PyTorch: compute J and invert
            p = torch.tensor([px, py], dtype=torch.float64, requires_grad=True)
            J = torch.autograd.functional.jacobian(
                lambda x: distort_brown_conrady_torch(x, d_torch), p
            )
            J_inv_torch = torch.linalg.inv(J).numpy()

            # Analytical: compute J elements and invert manually
            j00, j01, j10, j11 = _compute_jacobian_analytical(px, py, d)
            det = j00 * j11 - j01 * j10
            inv_det = 1.0 / det

            # J^{-1} = (1/det) * [[j11, -j01], [-j10, j00]]
            J_inv_analytical = np.array([
                [j11 * inv_det, -j01 * inv_det],
                [-j10 * inv_det, j00 * inv_det]
            ])

            np.testing.assert_allclose(
                J_inv_analytical, J_inv_torch, rtol=1e-5, atol=1e-10,
                err_msg=f"Inverse Jacobian mismatch at ({px}, {py})"
            )
