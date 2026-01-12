"""Tests for Jacobian correctness."""

import numpy as np
import pytest

import deltacamera.validity
import deltacamera.distortion
from conftest import BROWN_CONRADY_COEFFS, extend_distortion_coeffs


class TestJacobianNumerical:
    """Test that analytical Jacobian matches numerical approximation."""

    @pytest.mark.parametrize("d", BROWN_CONRADY_COEFFS[:2])
    def test_jacobian_determinant_matches_numerical(self, d):
        """Analytical Jacobian determinant should match finite differences."""
        points = np.array([
            [0.01, 0.01],  # Avoid exact origin for numerical stability
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.2],
            [-0.15, 0.1],
            [0.2, -0.1],
        ], np.float32)

        # Convert to polar for analytical Jacobian
        r = np.linalg.norm(points, axis=1).astype(np.float32)
        t = np.arctan2(points[:, 1], points[:, 0]).astype(np.float32)

        # Analytical Jacobian determinant (in polar coordinates)
        analytical_det = deltacamera.validity.jacobian_det_polar(r, t, d)

        # Numerical Jacobian via finite differences (in Cartesian)
        eps = 1e-5
        numerical_det = np.zeros(len(points), np.float32)

        for i, p in enumerate(points):
            # Compute partial derivatives numerically
            p_xp = deltacamera.distortion.distort_points(
                p.reshape(1, 2) + [[eps, 0]], d, check_validity=False
            )[0]
            p_xm = deltacamera.distortion.distort_points(
                p.reshape(1, 2) - [[eps, 0]], d, check_validity=False
            )[0]
            p_yp = deltacamera.distortion.distort_points(
                p.reshape(1, 2) + [[0, eps]], d, check_validity=False
            )[0]
            p_ym = deltacamera.distortion.distort_points(
                p.reshape(1, 2) - [[0, eps]], d, check_validity=False
            )[0]

            dx_dx = (p_xp[0] - p_xm[0]) / (2 * eps)
            dx_dy = (p_yp[0] - p_ym[0]) / (2 * eps)
            dy_dx = (p_xp[1] - p_xm[1]) / (2 * eps)
            dy_dy = (p_yp[1] - p_ym[1]) / (2 * eps)

            numerical_det[i] = dx_dx * dy_dy - dx_dy * dy_dx

        # Tolerance increased slightly due to float32 finite difference precision
        np.testing.assert_allclose(analytical_det, numerical_det, rtol=2e-3, atol=1e-6)

    def test_jacobian_at_origin_is_one(self):
        """Jacobian determinant at origin should be 1."""
        for d in BROWN_CONRADY_COEFFS:
            r = np.array([0.0], np.float32)
            t = np.array([0.0], np.float32)
            det = deltacamera.validity.jacobian_det_polar(r, t, d)
            np.testing.assert_allclose(det, 1.0, rtol=1e-6)

    def test_jacobian_symmetry(self):
        """For radially symmetric distortion, Jacobian should be radially symmetric."""
        # Only radial coefficients (no tangential)
        d = extend_distortion_coeffs(np.array([-0.2, 0.1, 0, 0, 0.01], np.float32))

        # Points at same radius, different angles
        radius = 0.3
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False).astype(np.float32)
        r = np.full_like(angles, radius)

        det = deltacamera.validity.jacobian_det_polar(r, angles, d)

        # All should have same determinant
        np.testing.assert_allclose(det, det[0], rtol=1e-5)
