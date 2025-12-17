"""Shared fixtures and helpers for lensform tests."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import lensform


# =============================================================================
# Sample distortion coefficients from real calibrations
# =============================================================================

# Brown-Conrady 12-parameter model: k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4
BROWN_CONRADY_COEFFS = [
    # Moderate barrel distortion
    np.array([-0.336, 0.160, 1.27e-4, -7.23e-5, -0.046, 0, 0, 0, 0, 0, 0, 0], np.float32),
    # Pincushion distortion
    np.array([0.15, -0.08, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0], np.float32),
    # With rational coefficients (k4, k5, k6)
    np.array([0.1, -0.2, 0.001, -0.001, 0.05, 0.02, -0.01, 0.005, 0, 0, 0, 0], np.float32),
    # Mild distortion
    np.array([-0.05, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.float32),
]

# 5-parameter Brown-Conrady (OpenCV default): k1, k2, p1, p2, k3
BROWN_CONRADY_5_COEFFS = [
    np.array([-0.336, 0.160, 1.27e-4, -7.23e-5, -0.046], np.float32),
    np.array([0.1, -0.05, 0, 0, 0.01], np.float32),
]

# Fisheye (Kannala-Brandt) 4-parameter model: k1, k2, k3, k4
FISHEYE_COEFFS = [
    np.array([0.1, -0.2, 0.05, -0.01], np.float32),
    np.array([-0.05, 0.03, -0.01, 0.002], np.float32),
    np.array([0.2, 0.1, 0.05, 0.02], np.float32),
]

# Sample intrinsic matrices
INTRINSIC_MATRICES = [
    np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float32),  # Square pixels
    np.array([[600, 0, 400], [0, 600, 300], [0, 0, 1]], np.float32),  # Different size
    np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], np.float32),  # HD
]


# =============================================================================
# Helper functions
# =============================================================================

def sample_inside_polygon(vertices, n=100, margin=0.0):
    """Sample random points inside a polygon.

    Args:
        vertices: (N, 2) array of polygon vertices
        n: number of points to sample
        margin: shrink polygon by this factor (0.1 = 10% margin from boundary)

    Returns:
        (n, 2) array of points inside the polygon
    """
    from shapely.geometry import Polygon, Point

    poly = Polygon(vertices)
    if margin > 0:
        poly = poly.buffer(-margin * np.sqrt(poly.area))
        if poly.is_empty:
            poly = Polygon(vertices).buffer(-0.01 * np.sqrt(Polygon(vertices).area))

    minx, miny, maxx, maxy = poly.bounds

    points = []
    max_attempts = n * 100
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if poly.contains(p):
            points.append([p.x, p.y])
        attempts += 1

    if len(points) < n:
        raise ValueError(f"Could only sample {len(points)}/{n} points inside polygon")

    return np.array(points, dtype=np.float32)


def sample_on_polygon_boundary(vertices, n=100):
    """Sample random points on a polygon boundary."""
    from shapely.geometry import Polygon

    poly = Polygon(vertices)
    boundary = poly.exterior

    points = []
    for _ in range(n):
        t = np.random.uniform(0, boundary.length)
        p = boundary.interpolate(t)
        points.append([p.x, p.y])

    return np.array(points, dtype=np.float32)


def random_rotation_matrix():
    """Generate a random rotation matrix."""
    return Rotation.random().as_matrix().astype(np.float32)


def small_rotation_matrix(angle_deg=5):
    """Generate a small rotation matrix around a random axis."""
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    return Rotation.from_rotvec(axis * angle_rad).as_matrix().astype(np.float32)


def extend_distortion_coeffs(d, target_len=12):
    """Extend distortion coefficients to target length with zeros."""
    if len(d) >= target_len:
        return d[:target_len].astype(np.float32)
    result = np.zeros(target_len, np.float32)
    result[:len(d)] = d
    return result


def make_camera_with_distortion(distortion_coeffs, intrinsic=None):
    """Create a camera with specified distortion coefficients."""
    if intrinsic is None:
        intrinsic = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
    return lensform.Camera(
        intrinsic_matrix=intrinsic,
        distortion_coeffs=distortion_coeffs,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_camera():
    """A simple pinhole camera with no distortion."""
    return lensform.Camera(
        intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
    )


@pytest.fixture
def distorted_camera():
    """A camera with Brown-Conrady distortion."""
    return lensform.Camera(
        intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
        distortion_coeffs=BROWN_CONRADY_5_COEFFS[0],
    )


@pytest.fixture
def fisheye_camera():
    """A camera with fisheye distortion."""
    return lensform.Camera(
        intrinsic_matrix=[[300, 0, 320], [0, 300, 240], [0, 0, 1]],
        distortion_coeffs=FISHEYE_COEFFS[0],
    )


@pytest.fixture
def positioned_camera():
    """A camera with position and rotation."""
    return lensform.Camera(
        optical_center=[1, 2, 3],
        rot_world_to_cam=random_rotation_matrix(),
        intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
    )
