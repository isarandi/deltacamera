"""Shared fixtures and helpers for deltacamera tests."""

import os
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import deltacamera


# =============================================================================
# Helper: extend distortion coefficients (defined early for use in loading)
# =============================================================================

def extend_distortion_coeffs(d, target_len=12):
    """Extend distortion coefficients to target length with zeros."""
    if len(d) >= target_len:
        return d[:target_len].astype(np.float32)
    result = np.zeros(target_len, np.float32)
    result[:len(d)] = d
    return result


# =============================================================================
# Load real calibration data
# =============================================================================

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
_DNA_RENDERING_PATH = os.path.join(_DATA_DIR, 'dna_rendering_calibrations.npz')
_JRDB_PATH = os.path.join(_DATA_DIR, 'jrdb_calibrations.npz')
_FISHEYE_PATH = os.path.join(_DATA_DIR, 'egohumans_fisheye_calibrations.npz')

def _load_brown_conrady_calibrations():
    """Load real Brown-Conrady calibrations from DNA-Rendering and JRDB datasets."""
    all_distortion = []
    all_intrinsics = []
    all_types = []

    # DNA-Rendering (high resolution, stronger distortion)
    if os.path.exists(_DNA_RENDERING_PATH):
        data = np.load(_DNA_RENDERING_PATH, allow_pickle=True)
        all_distortion.extend(data['distortion_coeffs'])
        all_intrinsics.extend(data['intrinsic_matrices'])
        all_types.extend(data['camera_types'])

    # JRDB (lower resolution, moderate distortion)
    if os.path.exists(_JRDB_PATH):
        data = np.load(_JRDB_PATH, allow_pickle=True)
        all_distortion.extend(data['distortion_coeffs'])
        all_intrinsics.extend(data['intrinsic_matrices'])
        all_types.extend(data['camera_names'])

    if all_distortion:
        return {
            'distortion_coeffs': all_distortion,
            'intrinsic_matrices': all_intrinsics,
            'camera_types': all_types,
        }
    return None

def _load_fisheye_calibrations():
    """Load real fisheye calibrations from EgoHumans dataset."""
    if os.path.exists(_FISHEYE_PATH):
        data = np.load(_FISHEYE_PATH, allow_pickle=True)
        return {
            'distortion_coeffs': data['distortion_coeffs'],
            'intrinsic_matrices': data['intrinsic_matrices'],
            'camera_types': data['camera_types'],
        }
    return None

_REAL_CALIBRATIONS = _load_brown_conrady_calibrations()
_FISHEYE_CALIBRATIONS = _load_fisheye_calibrations()


# =============================================================================
# Distortion coefficients for testing
# =============================================================================

# Real calibrations from DNA-Rendering (5-param OpenCV: k1, k2, p1, p2, k3)
# Extended to 12-param Brown-Conrady format for compatibility
if _REAL_CALIBRATIONS is not None:
    BROWN_CONRADY_COEFFS = [
        extend_distortion_coeffs(d) for d in _REAL_CALIBRATIONS['distortion_coeffs']
    ]
    BROWN_CONRADY_5_COEFFS = list(_REAL_CALIBRATIONS['distortion_coeffs'])
    INTRINSIC_MATRICES = list(_REAL_CALIBRATIONS['intrinsic_matrices'])
else:
    # Fallback synthetic data if real calibrations not available
    BROWN_CONRADY_COEFFS = [
        np.array([-0.336, 0.160, 1.27e-4, -7.23e-5, -0.046, 0, 0, 0, 0, 0, 0, 0], np.float32),
        np.array([0.15, -0.08, 0, 0, 0.02, 0, 0, 0, 0, 0, 0, 0], np.float32),
        np.array([0.1, -0.2, 0.001, -0.001, 0.05, 0.02, -0.01, 0.005, 0, 0, 0, 0], np.float32),
        np.array([-0.05, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.float32),
    ]
    BROWN_CONRADY_5_COEFFS = [
        np.array([-0.336, 0.160, 1.27e-4, -7.23e-5, -0.046], np.float32),
        np.array([0.1, -0.05, 0, 0, 0.01], np.float32),
    ]
    INTRINSIC_MATRICES = [
        np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float32),
        np.array([[600, 0, 400], [0, 600, 300], [0, 0, 1]], np.float32),
        np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], np.float32),
    ]

# Fisheye (Kannala-Brandt) 4-parameter model: k1, k2, k3, k4
# Real calibrations from EgoHumans dataset (Aria egocentric + exo cameras)
if _FISHEYE_CALIBRATIONS is not None:
    FISHEYE_COEFFS = list(_FISHEYE_CALIBRATIONS['distortion_coeffs'])
    FISHEYE_INTRINSIC_MATRICES = list(_FISHEYE_CALIBRATIONS['intrinsic_matrices'])
else:
    # Fallback synthetic data
    FISHEYE_COEFFS = [
        np.array([0.1, -0.2, 0.05, -0.01], np.float32),
        np.array([-0.05, 0.03, -0.01, 0.002], np.float32),
        np.array([0.2, 0.1, 0.05, 0.02], np.float32),
    ]
    FISHEYE_INTRINSIC_MATRICES = [
        np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]], np.float32),
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


def make_camera_with_distortion(distortion_coeffs, intrinsic=None):
    """Create a camera with specified distortion coefficients."""
    if intrinsic is None:
        intrinsic = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
    return deltacamera.Camera(
        intrinsic_matrix=intrinsic,
        distortion_coeffs=distortion_coeffs,
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_camera():
    """A simple pinhole camera with no distortion."""
    return deltacamera.Camera(
        intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
    )


@pytest.fixture
def distorted_camera():
    """A camera with Brown-Conrady distortion.

    Uses matching intrinsic matrix when real calibrations are available,
    otherwise uses synthetic data with compatible intrinsics.
    """
    if _REAL_CALIBRATIONS is not None:
        # Use real distortion with its matching intrinsic matrix
        return deltacamera.Camera(
            intrinsic_matrix=INTRINSIC_MATRICES[0],
            distortion_coeffs=BROWN_CONRADY_5_COEFFS[0],
        )
    else:
        return deltacamera.Camera(
            intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            distortion_coeffs=BROWN_CONRADY_5_COEFFS[0],
        )


@pytest.fixture
def fisheye_camera():
    """A camera with fisheye distortion.

    Uses matching intrinsic matrix when real calibrations are available.
    """
    if _FISHEYE_CALIBRATIONS is not None:
        return deltacamera.Camera(
            intrinsic_matrix=FISHEYE_INTRINSIC_MATRICES[0],
            distortion_coeffs=FISHEYE_COEFFS[0],
        )
    else:
        return deltacamera.Camera(
            intrinsic_matrix=[[300, 0, 320], [0, 300, 240], [0, 0, 1]],
            distortion_coeffs=FISHEYE_COEFFS[0],
        )


@pytest.fixture
def positioned_camera():
    """A camera with position and rotation."""
    return deltacamera.Camera(
        optical_center=[1, 2, 3],
        rot_world_to_cam=random_rotation_matrix(),
        intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
    )
