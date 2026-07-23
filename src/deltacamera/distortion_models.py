"""Lens distortion model classes for explicit model specification.

This module provides typed classes for specifying lens distortion models,
replacing the previous implicit detection based on coefficient array length.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import scipy.linalg


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
        new_R = np.array(R, copy=True)
        new_R[0] *= -1

        new_K = np.array(K, copy=True)
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
            angle: Rotation angle in radians (clockwise)
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
        new_K = np.array(K, copy=True)
        v = rot_image[1, :] @ new_K[:2, :2]
        v /= np.linalg.norm(v)
        rot_normalized = np.array([[v[1], -v[0]], v], dtype=np.float32)

        new_K[:2, :2] = rot_image @ new_K[:2, :2] @ rot_normalized.T
        new_K[:2, 2] = rot_image @ (new_K[:2, 2] - anchor) + anchor

        new_R = np.array(R, copy=True)
        new_R[:2] = rot_normalized @ new_R[:2]

        new_coeffs = coeffs.copy()

        # Handle tilt (tau_x, tau_y) if present
        if len(new_coeffs) >= 14 and (new_coeffs[12] != 0 or new_coeffs[13] != 0):
            # The rotated tilted model is exactly representable in the 14-param family.
            # The tilt homography factors as T = P @ N with P affine (diagonal 2x2 part)
            # and N = Ry(-tau_y) @ Rx(-tau_x) a rotation, so the requirement
            #     K' @ T' @ Rz(phi) = R_img @ K @ T
            # (with the 12-param coefficients conjugated by the same Rz(phi), and phi the
            # normalized-frame rotation that keeps K'[1, 0] = 0) is the classic
            # camera-matrix factorization: RQ-decompose the right-hand side into
            # upper-triangular times rotation, read tau' and phi off the rotation via
            # Y-X-Z Euler angles, and the affine factor is absorbed into K'.
            def tilt_matrix(tau_x, tau_y):
                cx, sx = np.cos(tau_x), np.sin(tau_x)
                cy, sy = np.cos(tau_y), np.sin(tau_y)
                return np.array(
                    [[cx, 0.0, 0.0], [-sx * sy, cy, 0.0], [sy, -cy * sx, cy * cx]]
                )

            angle64 = np.float64(angle)
            G = np.eye(3)
            G[:2, :2] = (
                np.array(
                    [
                        [np.cos(angle64), -np.sin(angle64)],
                        [np.sin(angle64), np.cos(angle64)],
                    ]
                )
                @ np.array(K[:2, :2], np.float64)
            )
            G = G @ tilt_matrix(np.float64(coeffs[12]), np.float64(coeffs[13]))

            U, M = scipy.linalg.rq(G)
            signs = np.sign(np.diag(U))
            M = M * signs[:, None]
            if np.linalg.det(M) < 0:
                M = -M

            # M = Ry(-tau_y') @ Rx(-tau_x') @ Rz(phi)
            tau_x_new = np.arcsin(M[1, 2])
            tau_y_new = -np.arctan2(M[0, 2], M[2, 2])
            phi = np.arctan2(M[1, 0], M[1, 1])

            new_coeffs[12] = np.float32(tau_x_new)
            new_coeffs[13] = np.float32(tau_y_new)
            rot_normalized = np.array(
                [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]], np.float32
            )
            rot_coeffs = rot_normalized
            Rz3 = np.array(
                [
                    [np.cos(phi), -np.sin(phi), 0.0],
                    [np.sin(phi), np.cos(phi), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            L = (G @ np.linalg.inv(tilt_matrix(tau_x_new, tau_y_new) @ Rz3))[:2, :2]
            new_K[:2, :2] = L.astype(np.float32)
            new_K[1, 0] = 0.0  # exactly zero by construction; clear the float residue
            new_R = np.array(R, copy=True)
            new_R[:2] = rot_normalized @ new_R[:2]
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
