"""Tests for vanishing point FOV estimation."""

import numpy as np
import pytest

from deltacamera import estimate_fov_from_vanishing_points, intrinsics_from_fov


class TestEstimateFovFromVanishingPoints:
    """Test estimate_fov_from_vanishing_points function."""

    def test_basic_orthogonal_vps(self):
        """Test with simple orthogonal vanishing points."""
        # Image center at ((1920-1)/2, (1080-1)/2) = (959.5, 539.5) for 1920x1080
        # Use VPs where dot product is negative:
        # VP1 at c + (1000, 100), VP2 at c + (-100, -100)
        # Dot = 1000*(-100) + 100*(-100) = -100000 - 10000 = -110000
        # f = sqrt(110000) = 331.66
        cx, cy = 959.5, 539.5
        result = estimate_fov_from_vanishing_points(
            vp1=[cx + 1000, cy + 100],
            vp2=[cx - 100, cy - 100],
            imshape=(1080, 1920)
        )

        assert result is not None
        assert 'f' in result
        assert 'fov_h' in result
        assert 'fov_v' in result
        assert 'fov_d' in result
        assert 'K' in result

        np.testing.assert_allclose(result['f'], np.sqrt(110000), rtol=1e-6)

    def test_returns_none_for_non_orthogonal_vps(self):
        """Test that non-orthogonal VPs return None."""
        # VPs that give positive dot product (non-orthogonal in 3D)
        # VP1 at (1500, 540), VP2 at (1400, 540)
        # Both on same horizontal line through center
        # (v1 - c) = (540, 0), (v2 - c) = (440, 0)
        # Dot = 540*440 + 0*0 = 237600 > 0
        result = estimate_fov_from_vanishing_points(
            vp1=[1500, 540],
            vp2=[1400, 540],
            imshape=(1080, 1920)
        )

        assert result is None

    def test_symmetric_vps(self):
        """Test with symmetric vanishing points around image center."""
        imshape = (1080, 1920)
        cx, cy = 959.5, 539.5

        # Symmetric VPs: one left of center, one above center
        # VP1 at (cx - d, cy + eps), VP2 at (cx + eps, cy - d)
        # This ensures negative dot product
        d = 1000
        eps = 100

        vp1 = [cx - d, cy + eps]  # Left of center
        vp2 = [cx + eps, cy - d]  # Above center

        # (v1 - c) = (-1000, 100), (v2 - c) = (100, -1000)
        # Dot = -1000*100 + 100*(-1000) = -100000 - 100000 = -200000
        # f = sqrt(200000) = 447.21

        result = estimate_fov_from_vanishing_points(vp1, vp2, imshape)

        assert result is not None
        np.testing.assert_allclose(result['f'], np.sqrt(200000), rtol=1e-6)

    def test_intrinsic_matrix_structure(self):
        """Test that returned intrinsic matrix has correct structure."""
        result = estimate_fov_from_vanishing_points(
            vp1=[1960, 640],
            vp2=[860, 440],
            imshape=(1080, 1920)
        )

        K = result['K']

        # K should be 3x3
        assert K.shape == (3, 3)

        # K should have structure [[f, 0, cx], [0, f, cy], [0, 0, 1]]
        assert K[0, 1] == 0  # No skew
        assert K[1, 0] == 0
        assert K[2, 0] == 0
        assert K[2, 1] == 0
        assert K[2, 2] == 1

        # fx == fy (square pixels assumed)
        np.testing.assert_allclose(K[0, 0], K[1, 1])

        # Principal point at image center
        np.testing.assert_allclose(K[0, 2], 959.5)
        np.testing.assert_allclose(K[1, 2], 539.5)

    def test_fov_consistency(self):
        """Test FOV values are consistent with focal length."""
        imshape = (1080, 1920)
        result = estimate_fov_from_vanishing_points(
            vp1=[1960, 640],
            vp2=[860, 440],
            imshape=imshape
        )

        f = result['f']
        h, w = imshape

        # Verify FOV formulas
        expected_fov_h = 2 * np.rad2deg(np.arctan(w / (2 * f)))
        expected_fov_v = 2 * np.rad2deg(np.arctan(h / (2 * f)))
        expected_fov_d = 2 * np.rad2deg(np.arctan(np.sqrt(w**2 + h**2) / (2 * f)))

        np.testing.assert_allclose(result['fov_h'], expected_fov_h, rtol=1e-6)
        np.testing.assert_allclose(result['fov_v'], expected_fov_v, rtol=1e-6)
        np.testing.assert_allclose(result['fov_d'], expected_fov_d, rtol=1e-6)

    def test_roundtrip_with_intrinsics_from_fov(self):
        """Test consistency with intrinsics_from_fov by roundtrip."""
        imshape = (1080, 1920)
        target_fov = 60.0

        # Create intrinsics for known FOV
        K = intrinsics_from_fov(target_fov, imshape)
        f = K[0, 0]
        cx, cy = K[0, 2], K[1, 2]

        # Construct VPs such that (v1-c)·(v2-c) = -f^2
        # Using: a = f, b = f, c = -f, d = 0
        # dot = f*(-f) + f*0 = -f^2
        vp1 = [cx + f, cy + f]
        vp2 = [cx - f, cy]

        result = estimate_fov_from_vanishing_points(vp1, vp2, imshape)

        assert result is not None
        # The recovered focal length should match (allowing for float32/64 precision)
        np.testing.assert_allclose(result['f'], f, rtol=1e-3)

    def test_handles_array_input(self):
        """Test that function handles numpy array input."""
        vp1 = np.array([1960.0, 640.0])
        vp2 = np.array([860.0, 440.0])
        imshape = (1080, 1920)

        result = estimate_fov_from_vanishing_points(vp1, vp2, imshape)

        assert result is not None
        assert isinstance(result['f'], (float, np.floating))

    def test_handles_3channel_imshape(self):
        """Test that function handles (H, W, C) image shape."""
        result = estimate_fov_from_vanishing_points(
            vp1=[1960, 640],
            vp2=[860, 440],
            imshape=(1080, 1920, 3)
        )

        assert result is not None
        # Should use only H, W
        np.testing.assert_allclose(result['K'][0, 2], 959.5)  # cx = (W-1)/2
        np.testing.assert_allclose(result['K'][1, 2], 539.5)  # cy = (H-1)/2

    def test_wide_fov(self):
        """Test estimation of wide FOV (>90 degrees)."""
        imshape = (1080, 1920)
        cx, cy = 959.5, 539.5

        # For wide FOV, focal length is small relative to image size
        # f = 400 gives h-fov ≈ 134 degrees
        f = 400

        # Construct VPs: dot = -f^2 = -160000
        # a = f = 400, b = f = 400, c = -f = -400, d = 0
        vp1 = [cx + f, cy + f]
        vp2 = [cx - f, cy]

        result = estimate_fov_from_vanishing_points(vp1, vp2, imshape)

        assert result is not None
        assert result['fov_h'] > 90  # Wide angle

    def test_narrow_fov(self):
        """Test estimation of narrow FOV (<30 degrees)."""
        imshape = (1080, 1920)
        cx, cy = 959.5, 539.5

        # For narrow FOV, focal length is large
        # f = 4000 gives h-fov ≈ 27 degrees
        f = 4000

        # Construct VPs
        vp1 = [cx + f, cy + f]
        vp2 = [cx - f, cy]

        result = estimate_fov_from_vanishing_points(vp1, vp2, imshape)

        assert result is not None
        assert result['fov_h'] < 30  # Narrow angle
