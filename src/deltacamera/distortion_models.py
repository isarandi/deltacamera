"""Lens distortion model classes for explicit model specification.

This module provides typed classes for specifying lens distortion models,
replacing the previous implicit detection based on coefficient array length.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True, eq=False)
class LensDistortionModel:
    """Base class for lens distortion models.

    This is a frozen dataclass, making instances immutable and hashable.
    The coefficients array is locked (read-only) after construction.

    Attributes:
        coeffs: Distortion coefficients as a read-only float32 numpy array.
    """

    coeffs: np.ndarray

    def __post_init__(self):
        # Convert to float32 array and make a copy
        arr = np.asarray(self.coeffs, dtype=np.float32).copy()
        arr.flags.writeable = False
        # Use object.__setattr__ since this is a frozen dataclass
        object.__setattr__(self, "coeffs", arr)

    def __hash__(self) -> int:
        return hash(self.coeffs.tobytes())

    def __eq__(self, other) -> bool:
        if not isinstance(other, LensDistortionModel):
            return NotImplemented
        return type(self) is type(other) and np.array_equal(self.coeffs, other.coeffs)

    def has_nonzero_coeffs(self) -> bool:
        """Return True if any coefficient is non-zero."""
        return cv2.hasNonZero(self.coeffs)


@dataclass(frozen=True, eq=False)
class FisheyeKannalaBrandt(LensDistortionModel):
    """Kannala-Brandt fisheye distortion model.

    Requires exactly 4 coefficients (k1, k2, k3, k4).

    The model uses the equidistant projection with radial distortion:
        theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)

    where theta is the angle between the optical axis and the incoming ray.
    """

    def __post_init__(self):
        super().__post_init__()
        if len(self.coeffs) != 4:
            raise ValueError(
                f"FisheyeKannalaBrandt requires exactly 4 coefficients, got {len(self.coeffs)}"
            )


@dataclass(frozen=True, eq=False)
class BrownConradyEx(LensDistortionModel):
    """Brown-Conrady distortion model.

    Requires 5, 8, 12, or 14 coefficients.

    Coefficient layout:
        - 5 coeffs:  k1, k2, p1, p2, k3
        - 8 coeffs:  k1, k2, p1, p2, k3, k4, k5, k6
        - 12 coeffs: k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4
        - 14 coeffs: k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y

    Where:
        - k1-k6: Radial distortion coefficients
        - p1, p2: Tangential distortion coefficients
        - s1-s4: Thin prism distortion coefficients
        - tau_x, tau_y: Tilt angles for the tilted sensor model
    """

    def __post_init__(self):
        super().__post_init__()
        valid_lengths = {5, 8, 12, 14}
        if len(self.coeffs) not in valid_lengths:
            raise ValueError(
                f"BrownConradyEx requires 5, 8, 12, or 14 coefficients, got {len(self.coeffs)}"
            )

    @staticmethod
    def transform_for_hflip(
        R: np.ndarray, K: np.ndarray, coeffs: np.ndarray, imshape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform R, K, and distortion coeffs for horizontal image flip.

        Args:
            R: 3x3 rotation matrix (world-to-camera)
            K: 3x3 intrinsic matrix
            coeffs: Distortion coefficients
            imshape: Image shape (height, width)

        Returns:
            Tuple of (new_R, new_K, new_coeffs)
        """
        new_R = R.copy()
        new_R[0] *= -1

        new_K = K.copy()
        new_K[0, 2] = (imshape[1] - 1) - new_K[0, 2]
        new_K[0, 1] *= -1  # negate skew

        new_coeffs = coeffs.copy()
        new_coeffs[3] *= -1  # p2
        if len(new_coeffs) > 8:
            new_coeffs[[8, 9]] *= -1  # s1, s2
        if len(new_coeffs) >= 14:
            new_coeffs[13] *= -1  # tau_y

        return new_R, new_K, new_coeffs

    @staticmethod
    def transform_for_rotation(
        R: np.ndarray, K: np.ndarray, coeffs: np.ndarray, angle: float, anchor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform R, K, and distortion coeffs for image rotation.

        Args:
            R: 3x3 rotation matrix (world-to-camera)
            K: 3x3 intrinsic matrix
            coeffs: Distortion coefficients
            angle: Rotation angle in radians (counter-clockwise)
            anchor: Rotation center (x, y)

        Returns:
            Tuple of (new_R, new_K, new_coeffs)
        """
        angle = np.float32(angle)
        sin = np.sin(angle)
        cos = np.cos(angle)
        rot_image = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)

        # Compute rot_normalized: rotation in normalized coords that avoids K[1,0] skew
        # See docs/explanation/distortion-rotation.rst for derivation
        new_K = K.copy()
        v = rot_image[1, :] @ new_K[:2, :2]
        v /= np.linalg.norm(v)
        rot_normalized = np.array([[v[1], -v[0]], v], dtype=np.float32)

        new_K[:2, :2] = rot_image @ new_K[:2, :2] @ rot_normalized.T
        new_K[:2, 2] = rot_image @ (new_K[:2, 2] - anchor) + anchor

        new_R = R.copy()
        new_R[:2] = rot_normalized @ new_R[:2]

        new_coeffs = coeffs.copy()

        # Handle tilt (tau_x, tau_y) if present - requires Euler angle reordering
        if len(new_coeffs) >= 14 and (new_coeffs[12] != 0 or new_coeffs[13] != 0):
            # Reorder Euler angles: XYZ -> ZXY to get new tau values
            angle_normalized = np.arctan2(v[0], v[1])
            angle_coeffs, new_coeffs[12], new_coeffs[13] = Rotation.from_euler(
                'xyz', [coeffs[12], coeffs[13], angle_normalized]
            ).as_euler('zxy')

            # rot_coeffs is the rotation to apply to tangential/thin-prism coefficients
            rot_coeffs = Rotation.from_euler('z', angle_coeffs).as_matrix()[:2, :2]
            rot_coeffs = rot_coeffs.astype(np.float32)

            # Apply tilt homography correction to intrinsics
            tilt_rot = Rotation.from_euler('xy', [coeffs[12], coeffs[13]]).as_matrix()
            tilt_homography = np.array([
                [tilt_rot[2, 2], 0, -tilt_rot[0, 2]],
                [0, tilt_rot[2, 2], -tilt_rot[1, 2]],
                [0, 0, 1]], dtype=np.float32)

            rot_normalized_3x3 = np.array([
                [rot_normalized[0, 0], rot_normalized[0, 1], 0],
                [rot_normalized[1, 0], rot_normalized[1, 1], 0],
                [0, 0, 1]], dtype=np.float32)

            tilt_homography_rotated = rot_normalized_3x3 @ tilt_homography @ rot_normalized_3x3.T

            tilt_rot_new = Rotation.from_euler('xy', [new_coeffs[12], new_coeffs[13]]).as_matrix()
            tilt_scale = 1 / tilt_rot_new[2, 2]
            tilt_homography_new_inv = np.array([
                [tilt_scale, 0, tilt_rot_new[0, 2] * tilt_scale],
                [0, tilt_scale, tilt_rot_new[1, 2] * tilt_scale],
                [0, 0, 1]], dtype=np.float32)

            new_K = new_K @ tilt_homography_rotated @ tilt_homography_new_inv
        else:
            # No tilt: use rot_normalized for tangential/thin-prism rotation
            rot_coeffs = rot_normalized

        # Rotate tangential (p1, p2) and thin prism (s1-s4)
        new_coeffs[[[3], [2]]] = rot_coeffs @ new_coeffs[[[3], [2]]]
        if len(new_coeffs) > 8:
            new_coeffs[[[8, 9], [10, 11]]] = rot_coeffs @ new_coeffs[[[8, 9], [10, 11]]]

        return new_R, new_K, new_coeffs


def infer_distortion_model(
    coeffs: Optional[np.ndarray],
) -> Optional[LensDistortionModel]:
    """Infer the distortion model type from coefficient array.

    Args:
        coeffs: Distortion coefficient array, or None.

    Returns:
        - None if coeffs is None or all zeros
        - FisheyeKannalaBrandt if exactly 4 coefficients
        - BrownConradyEx for other lengths (5, 8, 12, or 14)

    Raises:
        ValueError: If coefficient length is invalid for Brown-Conrady model.
    """
    if coeffs is None:
        return None

    coeffs = np.asarray(coeffs, dtype=np.float32)
    if not cv2.hasNonZero(coeffs):
        return None

    if len(coeffs) == 4:
        return FisheyeKannalaBrandt(coeffs)
    else:
        return BrownConradyEx(coeffs)
