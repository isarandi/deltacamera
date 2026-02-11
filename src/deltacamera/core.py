from typing import Optional, Sequence, TYPE_CHECKING, Union
import warnings

import cv2
import numpy as np
import shapely
from scipy.spatial.transform import Rotation

from . import coordframes, distortion, validity
from .decorators import DeprecatingArray, camera_transform, point_transform
from .distortion_models import (
    BrownConradyEx,
    FisheyeKannalaBrandt,
    LensDistortionModel,
    infer_distortion_model,
)
from .util import allclose_or_nones, equal_or_nones, unit_vec

if TYPE_CHECKING:
    from . import Camera


class Camera:
    """Pinhole camera with extrinsic and intrinsic calibration with optional distortions.

    The camera coordinate system has the following axes:
      x points to the right
      y points down
      z points forwards

    The world z direction is assumed to point up by default, but `world_up` can also be
     specified differently.

    Args:
        optical_center: position of the camera in world coordinates (eye point)
        rot_world_to_cam: 3x3 rotation matrix for transforming column vectors
            from being expressed in world reference frame to being expressed in camera
            reference frame as follows:
            column_point_cam = rot_matrix_world_to_cam @ (column_point_world - optical_center)
        intrinsic_matrix: 3x3 matrix that maps 3D points in camera space to homogeneous
            coordinates in image (pixel) space. Its last row must be (0,0,1).
        distortion_coeffs: parameters describing radial and tangential lens distortions,
            following OpenCV's model and order: k1, k2, p1, p2, k3 or None,
            if the camera has no distortion.
        world_up: a world vector that is designated as "pointing up".
        extrinsic_matrix: 4x4 extrinsic transformation matrix as an alternative to
            providing `optical_center` and `rot_world_to_cam`.
        trans_after_rot: translation vector to apply after the rotation
            (alternative to optical_center, which is a negative translation before the rotation)
    """

    def __init__(
        self,
        optical_center=None,
        rot_world_to_cam=None,
        intrinsic_matrix=np.eye(3),
        distortion_coeffs=None,
        world_up=(0, 0, 1),
        extrinsic_matrix=None,
        trans_after_rot=None,
        image_shape=None,
        image_size=None,
        distortion_model: Optional[LensDistortionModel] = None,
    ):
        dtype = np.float32
        if optical_center is not None and extrinsic_matrix is not None:
            raise ValueError(
                "Provide only one of `optical_center`, `trans_after_rot` or `extrinsic_matrix`!"
            )
        if extrinsic_matrix is not None and rot_world_to_cam is not None:
            raise ValueError("Provide only one of `rot_world_to_cam` or `extrinsic_matrix`!")

        if (optical_center is None) and (trans_after_rot is None) and (extrinsic_matrix is None):
            optical_center = np.zeros(3, dtype=dtype)

        if (rot_world_to_cam is None) and (extrinsic_matrix is None):
            rot_world_to_cam = np.eye(3, dtype=dtype)

        if extrinsic_matrix is not None:
            self.extrinsic_matrix = np.asanyarray(extrinsic_matrix, dtype=dtype).view(DeprecatingArray)
        else:
            self.R = np.asanyarray(rot_world_to_cam, dtype=dtype).view(DeprecatingArray)
            if optical_center is not None:
                self.t = np.asanyarray(optical_center, dtype=dtype).view(DeprecatingArray)
            else:
                self.t = (-self.R.T @ np.asarray(trans_after_rot, dtype=dtype)).view(DeprecatingArray)

        self.intrinsic_matrix = np.asanyarray(intrinsic_matrix, dtype=dtype).view(DeprecatingArray)

        # Handle distortion: prefer distortion_model, deprecate distortion_coeffs
        if distortion_coeffs is not None and distortion_model is not None:
            raise ValueError(
                "Provide only one of `distortion_coeffs` or `distortion_model`, not both."
            )

        if distortion_coeffs is not None:
            warnings.warn(
                "distortion_coeffs parameter is deprecated. "
                "Use distortion_model=BrownConradyEx(...) or distortion_model=FisheyeKannalaBrandt(...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._distortion_model = infer_distortion_model(distortion_coeffs)
        else:
            self._distortion_model = distortion_model

        self.world_up = np.asanyarray(world_up, dtype=dtype).copy()
        self.world_up /= np.linalg.norm(self.world_up)
        self.world_up = self.world_up.view(DeprecatingArray)

        # Handle image_shape / image_size (mutually exclusive)
        if image_shape is not None and image_size is not None:
            raise ValueError("Provide only one of `image_shape` or `image_size`, not both.")

        if image_size is not None:
            self._image_shape = (int(image_size[1]), int(image_size[0]))
        elif image_shape is not None:
            self._image_shape = (int(image_shape[0]), int(image_shape[1]))
        else:
            self._image_shape = None

        if not np.allclose(self.intrinsic_matrix[2, :], [0, 0, 1]):
            raise ValueError(
                f"Bottom row of intrinsic matrix must be (0,0,1), "
                f"got {self.intrinsic_matrix[2, :]}."
            )
        if not np.isclose(self.intrinsic_matrix[1, 0], 0):
            raise ValueError(
                f"Skew of y (intr[1,0]) must be zero, got {self.intrinsic_matrix[1, 0]}."
            )

    # Methods to transform between coordinate systems (world, camera, image)
    @point_transform
    def camera_to_image(
        self,
        points: np.ndarray,
        validate_distortion: bool = True,
        dst: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform points from 3D camera coordinate space to image space.
        The steps involved are

        1. Projection
        2. Lens distortion
        3. Intrinsic matrix (focal length and principal point, possibly skew)

        Args:
            points: points in camera coordinates

        Returns:
            points in image coordinates
        """

        if not self.has_distortion():
            if points.shape[1] == 3:
                return coordframes.project_and_apply_intrinsics(
                    points, self.intrinsic_matrix, dst=dst
                )
            else:
                return coordframes.apply_intrinsics(
                    points, self.intrinsic_matrix, dst=dst
                )

        if points.shape[1] == 3:
            pun = coordframes.project(points, dst=dst)
            pn_dst = pun
        else:
            pun = points
            pn_dst = dst

        if self.has_fisheye_distortion():
            pn = distortion.distort_points_fisheye(
                pun, self._distortion_model.coeffs, check_validity=validate_distortion, dst=pn_dst
            )
        else:
            pn = distortion.distort_points(
                pun, self.get_distortion_coeffs(12), check_validity=validate_distortion, dst=pn_dst
            )
        return coordframes.apply_intrinsics(pn, self.intrinsic_matrix, dst=pn)

    @point_transform
    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """Transform points from world coordinate space to camera coordinate space.

        Args:
            points: points in world coordinates

        Returns:
            points in camera coordinates
        """
        return coordframes.world_to_camera(points, self.R, self.t)

    @point_transform
    def camera_to_world(self, points: np.ndarray) -> np.ndarray:
        """Transform points from camera coordinate space to world coordinate space.

        Args:
            points: points in camera coordinates

        Returns:
            points in world coordinates
        """
        return coordframes.camera_to_world(points, self.R, self.t, dst=None)

    @point_transform
    def world_to_image(self, points: np.ndarray, validate_distortion: bool = True) -> np.ndarray:
        """Transform points from world coordinate space to image space.

        Args:
            points: points in world coordinates

        Returns:
            points in image coordinates
        """
        if not self.has_distortion():
            return coordframes.world_to_image(
                points, self.intrinsic_matrix, self.R, self.t
            )

        pun = coordframes.world_to_undist(points, self.R, self.t)
        if self.has_fisheye_distortion():
            pn = distortion.distort_points_fisheye(
                pun, self._distortion_model.coeffs, check_validity=validate_distortion, dst=pun
            )
        else:
            pn = distortion.distort_points(
                pun, self.get_distortion_coeffs(12), check_validity=validate_distortion, dst=pun
            )
        return coordframes.apply_intrinsics(pn, self.intrinsic_matrix, dst=pn)

    @point_transform
    def image_to_camera(
        self,
        points: np.ndarray,
        depth: Optional[Union[float, np.ndarray]] = 1.0,
        validate_distortion: bool = True,
        dst: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform points from image space to camera space.

        Args:
            points: points in image coordinates
            depth: depth of the points in camera space

        Returns:
            points in camera coordinates
        """
        if not self.has_distortion():
            if depth is None:
                return coordframes.undo_intrinsics(
                    points, self.intrinsic_matrix, dst=dst
                )
            elif np.isscalar(depth):
                return coordframes.backproject_K_depthval(
                    points, self.intrinsic_matrix, np.float32(depth), dst=dst
                )
            else:
                depth = depth.reshape(-1).astype(np.float32)
                return coordframes.backproject_K_deptharr(
                    points, self.intrinsic_matrix, depth, dst=dst
                )

        pn = coordframes.undo_intrinsics(points, self.intrinsic_matrix, dst=None)
        if self.has_fisheye_distortion():
            pun = distortion.undistort_points_fisheye(
                pn,
                self._distortion_model.coeffs,
                check_validity=validate_distortion,
            )
        else:
            pun = distortion.undistort_points(
                pn,
                self.get_distortion_coeffs(12),
                check_validity=validate_distortion,
            )

        if depth is None:
            if dst is not None:
                dst[:] = pun
                return dst
            else:
                return pun
        elif np.isscalar(depth):
            if depth == 1.0:
                return coordframes.backproject_homogeneous(pun, dst=dst)
            else:
                return coordframes.backproject_depthval(pun, np.float32(depth), dst=dst)
        else:
            depth = depth.reshape(-1).astype(np.float32)
            return coordframes.backproject_deptharr(pun, depth, dst=dst)

    @point_transform
    def image_to_world(
        self, points: np.ndarray, camera_depth: Union[float, np.ndarray] = 1
    ) -> np.ndarray:
        """Transform points from image space to world space.

        Args:
            points: points in image coordinates
            camera_depth: depth of the points in camera space

        Returns:
            points in world coordinates
        """

        pcam = self.image_to_camera(points, camera_depth)
        return coordframes.camera_to_world(pcam, self.R, self.t, dst=pcam)

    @point_transform
    def is_visible(
        self, world_points: np.ndarray, imsize: Sequence[int], validate_distortion: bool = False
    ) -> np.ndarray:
        """Check if points in world coordinates are visible in the image.

        A point is considered visible if it projects within the image frame and in front of the
        camera.

        Args:
            world_points: points in world coordinates (num_points, 3)
            imsize: size of the image (width, height)

        Returns:
            boolean array indicating for each point whether it is visible
        """

        imsize = np.asarray(imsize)
        cam_points = self.world_to_camera(world_points)

        # Check if the point is in front of the camera
        is_valid = cam_points[..., 2] > 0
        #
        # if check_distortion and self.has_distortion():
        #     # Check if the point is within the distortion limits
        #     checker_func = (
        #         validity.are_points_in_valid_region_fisheye
        #         if self.has_fisheye_distortion()
        #         else validity.are_points_in_valid_region
        #     )
        #     is_valid[is_valid] = checker_func(
        #         from_homogeneous(cam_points[is_valid]), self.distortion_coeffs
        #     )

        im_points = self.camera_to_image(cam_points, validate_distortion=validate_distortion)
        # Use strict upper bound: pixels at exactly (width, height) are out of bounds.
        # cv2.inRange uses <=, so use nextafter to get the largest float < imsize.
        ub_x = np.nextafter(np.float32(imsize[0]), np.float32(0))
        ub_y = np.nextafter(np.float32(imsize[1]), np.float32(0))
        is_valid = is_valid & cv2.inRange(
            im_points[np.newaxis], lowerb=(0, 0), upperb=(float(ub_x), float(ub_y)),
        ).squeeze(0)
        return is_valid

    # Methods to transform the camera parameters
    @camera_transform
    def shift_image(self, offset):
        """Adjust intrinsics so that the projected image is shifted by `offset`.

        Args:
            offset: an (x, y) offset vector. Positive values mean that the resulting image will
                shift towards the right and down.
        """
        self.intrinsic_matrix[:2, 2] += offset

    @camera_transform
    def shift_to_desired(self, current_coords_of_the_point, target_coords_of_the_point):
        """Shift the principal point to move a specific point to a desired location in the image.

        Args:
            current_coords_of_the_point: current location of the point of interest in the image
            target_coords_of_the_point: desired location of the point of interest in the image
        """

        self.intrinsic_matrix[:2, 2] += target_coords_of_the_point - current_coords_of_the_point

    @camera_transform
    def reset_roll(self):
        """Roll the camera upright by turning along the optical axis to align the vertical image
        axis with the vertical world axis (world up vector), as much as possible.
        """
        x = unit_vec(np.cross(self.R[2], self.world_up))
        if not np.all(np.isfinite(x)):
            return
        self.R[0] = x
        self.R[1] = -np.cross(self.R[0], self.R[2])

    @camera_transform
    def orbit_around(self, world_point_pivot, angle_radians, axis="vertical"):
        """Rotate the camera around a vertical or horizontal axis passing through `world point` by
        `angle_radians`.

        Args:
            world_point_pivot: the world coordinates of the pivot point to turn around
            angle_radians: the amount to rotate
            axis: 'vertical' or 'horizontal'.
        """

        if axis == "vertical":
            axis = self.world_up
        else:
            lookdir = self.R[2]
            axis = unit_vec(np.cross(lookdir, self.world_up))

        rot_matrix = cv2.Rodrigues(axis * angle_radians)[0]
        # The eye position rotates simply as any point
        self.t = (rot_matrix @ (self.t - world_point_pivot)) + world_point_pivot

        # R is rotated by a transform expressed in world coords, so it (its inverse since its a
        # coord transform matrix, not a point transform matrix) is applied on the right.
        # (inverse = transpose for rotation matrices, they are orthogonal)
        self.R = self.R @ rot_matrix.T

    @camera_transform
    def rotate(self, yaw=0, pitch=0, roll=0):
        """Rotate this camera by yaw, pitch, roll Euler angles in radians,
        relative to the current camera frame."""
        camera_rotation = Rotation.from_euler(
            "YXZ", [yaw, pitch, roll]).as_matrix().astype(np.float32)

        # The coordinates rotate according to the inverse of how the camera itself rotates
        point_coordinate_rotation = camera_rotation.T
        self.R = point_coordinate_rotation @ self.R

    @camera_transform
    def rotate_image(self, angle, imshape=None, anchor=None):
        """Transform the camera such that the produces image will be rotated around its center
        by `angle` radians (counter-clockwise)."""
        angle = np.float32(angle)
        sin = np.sin(angle)
        cos = np.cos(angle)
        R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)

        x = R[1, :] @ self.intrinsic_matrix[:2, :2]
        x /= np.linalg.norm(x)
        R_ = np.array([[x[1], -x[0]], x], dtype=np.float32)

        if anchor is None:
            anchor = (np.array(imshape[::-1], np.float32) - 1) / 2
        self.intrinsic_matrix[:2, :2] = R @ self.intrinsic_matrix[:2, :2] @ R_.T
        self.intrinsic_matrix[:2, 2] = R @ (self.intrinsic_matrix[:2, 2] - anchor) + anchor

        if self.has_nonfisheye_distortion():
            # Create new distortion model with rotated coefficients
            coeffs = self._distortion_model.coeffs.copy()
            coeffs[[[3], [2]]] = R_ @ coeffs[[[3], [2]]]
            if coeffs.shape[0] > 8:
                coeffs[[[8, 9], [10, 11]]] = R_ @ coeffs[[[8, 9], [10, 11]]]
            self._distortion_model = BrownConradyEx(coeffs)

        self.R[:2] = R_ @ self.R[:2]

    @camera_transform
    def rotate_image90(self, imshape, k=1):
        k %= 4
        if k == 0:
            pass
        elif k == 1:
            a = (imshape[0] - 1) / 2
            self.rotate_image(np.pi / 2, imshape, anchor=(a, a))
        elif k == 2:
            self.rotate_image(np.pi, imshape)
        else:
            a = (imshape[1] - 1) / 2
            self.rotate_image(-np.pi / 2, imshape, anchor=(a, a))

    def has_fisheye_distortion(self):
        """Check if the camera has fisheye (Kannala-Brandt) distortion."""
        return isinstance(self._distortion_model, FisheyeKannalaBrandt)

    def has_nonfisheye_distortion(self):
        """Check if the camera has non-fisheye (Brown-Conrady) distortion."""
        return isinstance(self._distortion_model, BrownConradyEx)

    def get_pitch_roll(self):
        yaw, pitch, roll = Rotation.from_matrix(self.R).as_euler("YXZ").astype(np.float32)
        return pitch, roll

    @camera_transform
    def zoom(self, factor):
        """Zooms the camera (factor > 1 makes objects look larger),
        while keeping the principal point fixed (scaling anchor is the principal point)."""
        self.intrinsic_matrix[:2, :2] *= np.expand_dims(np.float32(factor), -1)

    @camera_transform
    def scale_output(self, factor):
        """Adjusts the camera such that the images become scaled by `factor`. It's a scaling with
        the origin as anchor point.
        The difference with `self.zoom` is that this method also moves the principal point,
        multiplying its coordinates by `factor`."""
        self.intrinsic_matrix[:2] *= np.expand_dims(np.float32(factor), -1)
        if self._image_shape is not None:
            self._image_shape = (
                int(self._image_shape[0] * factor),
                int(self._image_shape[1] * factor),
            )

    @camera_transform
    def undistort(
        self, alpha_balance=None, imshape=None, new_imshape=None, center_principal_point=False
    ):
        """Undistort the camera by removing lens distortion and optionally adjusting the intrinsic
        matrix.

        After undistorting, the image content will not be rectangular. To make it rectangular,
        we either need to crop, or expand the "canvas", and include some black areas.


        Args:
            alpha_balance: if 0, set the zoom level such that no black pixels need to be added
                as padding at the borders. This removes some of the known pixel values.
                If 1, set the zoom level such that the image content is maximally preserved, but
                some black areas will be added. Between 0 and 1, it's a smooth transition
                between the two. If None, the zoom level is not changed, the old intrinsic matrix
                is kept.
            imshape: the shape of the input image (height, width).
            new_imshape: the shape of the output image (height, width). If None, the output image
                will have the same shape as the input image.
            center_principal_point: if True, the principal point will be moved to the center of the
                image.

        """
        if alpha_balance is not None and self.has_distortion():
            imsize = tuple(imshape[:2][::-1])
            new_imsize = imsize if new_imshape is None else tuple(new_imshape[:2][::-1])
            if self.has_fisheye_distortion():
                self.intrinsic_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.intrinsic_matrix,
                    self._distortion_model.coeffs,
                    imsize,
                    np.eye(3),
                    new_size=new_imsize,
                    balance=alpha_balance,
                )
            else:
                self.intrinsic_matrix = cv2.getOptimalNewCameraMatrix(
                    self.intrinsic_matrix,
                    self._distortion_model.coeffs,
                    imsize,
                    alpha_balance,
                    new_imsize,
                    centerPrincipalPoint=center_principal_point,
                )[0]

        self._distortion_model = None
        if center_principal_point:
            new_imshape = new_imshape if new_imshape is not None else imshape
            self.center_principal_point(new_imshape)

    def undistort_precise(
        self,
        imshape_distorted=None,
        imshape_undistorted=None,
        alpha_balance=None,
        center_principal_point=False,
        inplace=True,
    ):
        cam = self if inplace else self.copy()
        if alpha_balance is None:
            cam._distortion_model = None
            cam.square_pixels()
            cam.intrinsic_matrix[0, 1] = 0
            return cam, None, None
        else:
            cam.intrinsic_matrix, box, poly = (
                validity.get_optimal_undistorted_intrinsics(
                    cam,
                    imshape_distorted,
                    imshape_undistorted,
                    alpha_balance,
                    center_principal_point,
                )
            )
            cam.distortion_coeffs = None
            return cam, box, poly

    @camera_transform
    def square_pixels(self):
        """Adjusts the intrinsic matrix such that the pixels correspond to squares on the
        image plane."""
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        fmean = 0.5 * (fx + fy)
        multiplier = np.array([[fmean / fx, 0, 0], [0, fmean / fy, 0], [0, 0, 1]], np.float32)
        self.intrinsic_matrix = multiplier @ self.intrinsic_matrix

    @camera_transform
    def horizontal_flip(self):
        """Flip the camera horizontally by negating the first row of the rotation matrix.

        The principal point remains in the same position.
        """
        self.R[0] *= -1

    @camera_transform
    def horizontal_flip_image(self, imshape):
        """Flip the camera horizontally by negating the first row of the rotation matrix,
        and adjusting the intrinsic matrix and distortion coeffs so that the resulting image content
        is flipped."""
        self.horizontal_flip()
        self.intrinsic_matrix[0, 2] = (imshape[1] - 1) - self.intrinsic_matrix[0, 2]
        self.intrinsic_matrix[0, 1] *= -1
        if self.has_nonfisheye_distortion():
            # Create new distortion model with flipped coefficients
            coeffs = self._distortion_model.coeffs.copy()
            coeffs[3] *= -1  # p2
            if len(coeffs) > 8:
                coeffs[[8, 9]] *= -1  # s1, s2
            if len(coeffs) >= 14:
                coeffs[13] *= -1  # tau_y
            self._distortion_model = BrownConradyEx(coeffs)

    @camera_transform
    def center_principal_point(self, imshape):
        """Adjusts the intrinsic matrix so that the principal point becomes located at the center
        of an image sized imshape (height, width)"""

        self.intrinsic_matrix[:2, 2] = np.float32([imshape[1] - 1, imshape[0] - 1]) / 2

    @camera_transform
    def shift_to_center(self, desired_center_image_point, imshape):
        """Shifts the principal point such that what's currently at `desired_center_image_point`
        will be shown in the image center of an image shaped `imshape`."""

        current_coords_of_the_point = desired_center_image_point
        target_coords_of_the_point = np.float32([imshape[1] - 1, imshape[0] - 1]) / 2
        self.intrinsic_matrix[:2, 2] += target_coords_of_the_point - current_coords_of_the_point

    @camera_transform
    def turn_towards(
        self, target_image_point=None, target_world_point=None, target_cam_point=None
    ):
        """Turns the camera so that its optical axis goes through a desired target point.
        It resets any roll or horizontal flip applied previously. The resulting camera
        will not have horizontal flip and will be upright (0 roll)."""

        # assert (target_image_point is None) != (target_world_point is None)
        if target_image_point is not None:
            target_world_point = self.image_to_world(target_image_point)
        elif target_cam_point is not None:
            target_world_point = self.camera_to_world(target_cam_point)

        new_z = unit_vec(target_world_point - self.t)
        new_x = unit_vec(np.cross(new_z, self.world_up))
        if not np.all(np.isfinite(new_x)):
            # Looking along world_up, pick an arbitrary perpendicular frame
            new_x = unit_vec(np.cross(new_z, np.array([1, 0, 0], dtype=np.float32)))
            if not np.all(np.isfinite(new_x)):
                new_x = unit_vec(np.cross(new_z, np.array([0, 1, 0], dtype=np.float32)))
        new_y = np.cross(new_z, new_x)

        # row_stack because we need the inverse transform (we make a matrix that transforms
        # points from one coord system to another), which is the same as the transpose
        # for rotation matrices.
        self.R = np.row_stack([new_x, new_y, new_z]).astype(np.float32)

    # Getters
    def get_projection_matrix(self) -> np.ndarray:
        """Get the 3x4 projection matrix that maps 3D points in camera space to homogeneous
        coordinates in image space.
        This is only applicable if the camera has no distortion.
        """
        return coordframes.get_projection_matrix3x4(
            self.intrinsic_matrix, self.R, self.t
        )

    def get_extrinsic_matrix(self) -> np.ndarray:
        """Get the 4x4 extrinsic transformation matrix that maps 3D points in world space to
        3D points in camera space."""
        return coordframes.get_extrinsic_matrix(self.R, self.t)

    @property
    def extrinsic_matrix(self):
        return self.get_extrinsic_matrix()

    @extrinsic_matrix.setter
    def extrinsic_matrix(self, matrix: np.ndarray):
        self.R = matrix[:3, :3].astype(np.float32)
        self.t = -self.R.T @ matrix[:3, 3].astype(np.float32)

    def get_inv_extrinsic_matrix(self) -> np.ndarray:
        """Get the 4x4 extrinsic transformation matrix that maps 3D points in camera space to
        3D points in world space."""
        return coordframes.get_inv_extrinsic_matrix(self.R, self.t)

    def get_fov(self, imshape) -> float:
        """Get the field of view of the camera in degrees.

        This ignores the lens distortion coeffs."""
        focals = np.diagonal(self.intrinsic_matrix[:2, :2])
        # imshape is (height, width), focals is (fx, fy)
        # height goes with fy, width goes with fx, so reverse imshape
        return np.rad2deg(2 * np.arctan(np.max(imshape[:2][::-1] / (2 * focals))))

    def get_distortion_coeffs(
        self, n_coeffs_min: int = 5, n_coeffs_max: Optional[int] = None
    ) -> np.ndarray:
        """Get the distortion coefficients of the camera, padded or truncated as needed."""
        if self._distortion_model is None:
            return np.zeros(shape=(n_coeffs_min,), dtype=np.float32)
        coeffs = self._distortion_model.coeffs
        if self.has_fisheye_distortion():
            # Fisheye coefficients are returned as-is
            return coeffs
        elif len(coeffs) < n_coeffs_min:
            return np.pad(coeffs, (0, n_coeffs_min - len(coeffs)))
        elif n_coeffs_max is not None:
            return coeffs[:n_coeffs_max]
        else:
            return coeffs

    def has_distortion(self) -> bool:
        """Check if the camera has nonzero lens distortion."""
        return self._distortion_model is not None

    @property
    def distortion_model(self) -> Optional[LensDistortionModel]:
        """The lens distortion model, or None if undistorted."""
        return self._distortion_model

    @property
    def distortion_coeffs(self) -> Optional[np.ndarray]:
        """Get the raw distortion coefficients (deprecated).

        This property is deprecated. Use `distortion_model` instead.
        """
        warnings.warn(
            "distortion_coeffs property is deprecated. Use distortion_model instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._distortion_model is None:
            return None
        return self._distortion_model.coeffs

    @property
    def image_shape(self) -> Optional[tuple]:
        """Image shape as (height, width) tuple, or None if not set."""
        return self._image_shape

    def allclose(self, other_camera):
        """Check if all parameters of this camera are close to corresponding parameters
        of `other_camera`.

        Args:
            other_camera: the camera to compare to.

        Returns:
            True if all parameters are close, False otherwise.
        """
        # Get coeffs without triggering deprecation warning
        self_coeffs = self._distortion_model.coeffs if self._distortion_model else None
        other_coeffs = other_camera._distortion_model.coeffs if other_camera._distortion_model else None
        return (
            np.allclose(self.intrinsic_matrix, other_camera.intrinsic_matrix)
            and np.allclose(self.R, other_camera.R)
            and np.allclose(self.t, other_camera.t)
            and allclose_or_nones(self_coeffs, other_coeffs)
        )

    def is_equal(self, other):
        """Check if all parameters are exactly equal to another camera."""
        # Get coeffs without triggering deprecation warning
        self_coeffs = self._distortion_model.coeffs if self._distortion_model else None
        other_coeffs = other._distortion_model.coeffs if other._distortion_model else None
        return (
            np.array_equal(self.intrinsic_matrix, other.intrinsic_matrix)
            and np.array_equal(self.R, other.R)
            and np.array_equal(self.t, other.t)
            and equal_or_nones(self_coeffs, other_coeffs)
        )

    def __hash__(self) -> int:
        """Hash based on all camera parameters for use as dict key."""
        return hash((
            self.intrinsic_matrix.tobytes(),
            self.R.tobytes(),
            self.t.tobytes(),
            self.world_up.tobytes(),
            self._distortion_model,
            self._image_shape,
        ))

    def __eq__(self, other) -> bool:
        """Compare cameras for equality based on all parameters."""
        if not isinstance(other, Camera):
            return NotImplemented
        return (
            np.array_equal(self.intrinsic_matrix, other.intrinsic_matrix)
            and np.array_equal(self.R, other.R)
            and np.array_equal(self.t, other.t)
            and np.array_equal(self.world_up, other.world_up)
            and self._distortion_model == other._distortion_model
            and self._image_shape == other._image_shape
        )

    def __setstate__(self, state):
        """Wrap arrays as DeprecatingArray after unpickling and handle legacy pickles."""
        self.__dict__.update(state)

        # Wrap all arrays as DeprecatingArray
        self.intrinsic_matrix = np.asanyarray(self.intrinsic_matrix).view(DeprecatingArray)
        self.R = np.asanyarray(self.R).view(DeprecatingArray)
        self.t = np.asanyarray(self.t).view(DeprecatingArray)
        self.world_up = np.asanyarray(self.world_up).view(DeprecatingArray)

        # Handle legacy pickles that have distortion_coeffs instead of _distortion_model
        if "distortion_coeffs" in self.__dict__ and "_distortion_model" not in self.__dict__:
            self._distortion_model = infer_distortion_model(self.__dict__.pop("distortion_coeffs"))

        # Handle legacy pickles without _image_shape
        if "_image_shape" not in self.__dict__:
            self._image_shape = None

    def copy(
        self,
        *,
        intrinsic_matrix=None,
        rot_world_to_cam=None,
        R=None,
        optical_center=None,
        t=None,
        distortion_model=None,
        world_up=None,
        image_shape=None,
        image_size=None,
    ) -> "Camera":
        """Create a copy of this camera, optionally with modified parameters.

        Always creates a deep copy of arrays (structural sharing deferred to v1.0).

        Args:
            intrinsic_matrix: New intrinsic matrix (will be copied)
            rot_world_to_cam: New rotation matrix (will be copied)
            R: Alias for rot_world_to_cam
            optical_center: New optical center (will be copied)
            t: Alias for optical_center
            distortion_model: New distortion model
            world_up: New world up vector (will be copied)
            image_shape: New image shape (height, width)
            image_size: New image size (width, height)

        Returns:
            New Camera instance
        """
        dtype = np.float32

        # Handle aliases
        if R is not None and rot_world_to_cam is not None:
            raise ValueError("Provide only one of `R` or `rot_world_to_cam`")
        if R is not None:
            rot_world_to_cam = R
        if t is not None and optical_center is not None:
            raise ValueError("Provide only one of `t` or `optical_center`")
        if t is not None:
            optical_center = t

        # Handle image_shape/image_size
        if image_shape is not None and image_size is not None:
            raise ValueError("Provide only one of `image_shape` or `image_size`")
        if image_size is not None:
            image_shape = (int(image_size[1]), int(image_size[0]))

        # Create new camera bypassing __init__
        c = Camera.__new__(Camera)

        # Always deep copy arrays (structural sharing deferred to v1.0 when mutable API removed)
        # Intrinsic matrix
        if intrinsic_matrix is not None:
            c.intrinsic_matrix = np.asanyarray(intrinsic_matrix, dtype=dtype).copy().view(DeprecatingArray)
        else:
            c.intrinsic_matrix = np.array(self.intrinsic_matrix).view(DeprecatingArray)

        # Rotation matrix
        if rot_world_to_cam is not None:
            c.R = np.asanyarray(rot_world_to_cam, dtype=dtype).copy().view(DeprecatingArray)
        else:
            c.R = np.array(self.R).view(DeprecatingArray)

        # Translation / optical center
        if optical_center is not None:
            c.t = np.asanyarray(optical_center, dtype=dtype).copy().view(DeprecatingArray)
        else:
            c.t = np.array(self.t).view(DeprecatingArray)

        # World up
        if world_up is not None:
            wu = np.asanyarray(world_up, dtype=dtype).copy()
            wu /= np.linalg.norm(wu)
            c.world_up = wu.view(DeprecatingArray)
        else:
            c.world_up = np.array(self.world_up).view(DeprecatingArray)

        # Distortion model (immutable, always can be shared)
        if distortion_model is not None:
            c._distortion_model = distortion_model
        else:
            c._distortion_model = self._distortion_model

        # Image shape (immutable tuple, always can be shared)
        if image_shape is not None:
            c._image_shape = tuple(image_shape)
        else:
            c._image_shape = self._image_shape

        return c

    # =========================================================================
    # New Immutable API Methods (past-participle naming, return new Camera)
    # =========================================================================

    def zoomed(self, factor) -> "Camera":
        """Return a new camera with focal length scaled by factor.

        Args:
            factor: Zoom factor (>1 makes objects appear larger)

        Returns:
            New Camera with zoomed intrinsics
        """
        new_K = self.intrinsic_matrix.copy()
        new_K[:2, :2] *= np.expand_dims(np.float32(factor), -1)
        return self.copy(intrinsic_matrix=new_K)

    def rotated(self, yaw=0, pitch=0, roll=0) -> "Camera":
        """Return a new camera rotated by yaw, pitch, roll Euler angles in radians."""
        camera_rotation = Rotation.from_euler(
            "YXZ", [yaw, pitch, roll]).as_matrix().astype(np.float32)
        point_coordinate_rotation = camera_rotation.T
        new_R = point_coordinate_rotation @ self.R
        return self.copy(rot_world_to_cam=new_R)

    def turned_towards(self, target_point, up_vector=None) -> "Camera":
        """Return a new camera with optical axis pointing at target_point.

        Args:
            target_point: World coordinates of the target point
            up_vector: Optional world up vector (uses self.world_up if None)

        Returns:
            New Camera pointing at target
        """
        target_point = np.asarray(target_point, dtype=np.float32)
        world_up = np.asarray(up_vector, dtype=np.float32) if up_vector is not None else self.world_up

        new_z = unit_vec(target_point - self.t)
        new_x = unit_vec(np.cross(new_z, world_up))
        new_y = np.cross(new_z, new_x)

        new_R = np.row_stack([new_x, new_y, new_z]).astype(np.float32)
        return self.copy(rot_world_to_cam=new_R)

    def orbited_around(self, world_point, angle, axis="vertical") -> "Camera":
        """Return a new camera orbited around a world point.

        Args:
            world_point: World coordinates of the pivot point
            angle: Rotation angle in radians
            axis: 'vertical' or 'horizontal'

        Returns:
            New Camera at orbited position
        """
        world_point = np.asarray(world_point, dtype=np.float32)

        if axis == "vertical":
            axis_vec = self.world_up
        else:
            lookdir = self.R[2]
            axis_vec = unit_vec(np.cross(lookdir, self.world_up))

        rot_matrix = cv2.Rodrigues(axis_vec * angle)[0]
        new_t = (rot_matrix @ (self.t - world_point)) + world_point
        new_R = self.R @ rot_matrix.T

        return self.copy(rot_world_to_cam=new_R, optical_center=new_t)

    def rolled_upright(self) -> "Camera":
        """Return a new camera rolled upright to align with world up vector."""
        new_R = self.R.copy()
        new_x = unit_vec(np.cross(new_R[2], self.world_up))
        if not np.all(np.isfinite(new_x)):
            return self.copy()  # Can't roll if looking straight up/down
        new_R[0] = new_x
        new_R[1] = -np.cross(new_R[0], new_R[2])
        return self.copy(rot_world_to_cam=new_R)

    def hflipped(self) -> "Camera":
        """Return a new camera with horizontal flip (negated first row of rotation)."""
        new_R = self.R.copy()
        new_R[0] *= -1
        return self.copy(rot_world_to_cam=new_R)

    def image_shifted(self, offset) -> "Camera":
        """Return a new camera with principal point shifted by offset.

        Args:
            offset: (x, y) offset to shift the principal point

        Returns:
            New Camera with shifted principal point
        """
        offset = np.asarray(offset, dtype=np.float32)
        new_K = self.intrinsic_matrix.copy()
        new_K[:2, 2] += offset
        return self.copy(intrinsic_matrix=new_K)

    def point_shifted_to(self, current_point, target_point) -> "Camera":
        """Return a new camera with principal point adjusted to move a point.

        Args:
            current_point: Current image coordinates of a point
            target_point: Desired image coordinates of that point

        Returns:
            New Camera with adjusted principal point
        """
        current_point = np.asarray(current_point, dtype=np.float32)
        target_point = np.asarray(target_point, dtype=np.float32)
        offset = target_point - current_point
        return self.image_shifted(offset)

    def image_cropped(self, new_shape, anchor=(0, 0)) -> "Camera":
        """Return a camera adjusted for a cropped image.

        Args:
            new_shape: New image shape (height, width)
            anchor: Top-left corner of crop in original image (x, y)

        Returns:
            New Camera for the cropped image
        """
        anchor = np.asarray(anchor, dtype=np.float32)
        new_K = self.intrinsic_matrix.copy()
        new_K[0, 2] -= anchor[0]
        new_K[1, 2] -= anchor[1]
        return self.copy(intrinsic_matrix=new_K, image_shape=new_shape)

    def image_padded(self, new_shape, anchor=(0, 0)) -> "Camera":
        """Return a camera adjusted for a padded image.

        Args:
            new_shape: New image shape (height, width)
            anchor: Position of original image within padded image (x, y)

        Returns:
            New Camera for the padded image
        """
        anchor = np.asarray(anchor, dtype=np.float32)
        new_K = self.intrinsic_matrix.copy()
        new_K[0, 2] += anchor[0]
        new_K[1, 2] += anchor[1]
        return self.copy(intrinsic_matrix=new_K, image_shape=new_shape)

    def image_resized(self, new_shape) -> "Camera":
        """Return a camera adjusted for a resized image.

        Requires image_shape to be set to compute scale factors.

        Args:
            new_shape: New image shape (height, width)

        Returns:
            New Camera for the resized image
        """
        if self._image_shape is None:
            raise ValueError("image_shape must be set to resize camera")

        scale_y = new_shape[0] / self._image_shape[0]
        scale_x = new_shape[1] / self._image_shape[1]

        new_K = self.intrinsic_matrix.copy()
        new_K[0, :] *= scale_x
        new_K[1, :] *= scale_y

        return self.copy(intrinsic_matrix=new_K, image_shape=new_shape)

    def image_scaled(self, factor) -> "Camera":
        """Return a camera adjusted for a uniformly scaled image.

        Requires image_shape to be set.

        Args:
            factor: Scale factor (>1 makes image larger)

        Returns:
            New Camera for the scaled image
        """
        if self._image_shape is None:
            raise ValueError("image_shape must be set to scale camera")

        new_shape = (int(self._image_shape[0] * factor), int(self._image_shape[1] * factor))
        new_K = self.intrinsic_matrix.copy()
        new_K[:2] *= np.expand_dims(np.float32(factor), -1)

        return self.copy(intrinsic_matrix=new_K, image_shape=new_shape)

    def image_hflipped(self) -> "Camera":
        """Return a camera adjusted for a horizontally flipped image.

        Requires image_shape to be set.

        Returns:
            New Camera for the flipped image
        """
        if self._image_shape is None:
            raise ValueError("image_shape must be set for image_hflipped")

        if isinstance(self._distortion_model, BrownConradyEx):
            new_R, new_K, new_coeffs = BrownConradyEx.transform_for_hflip(
                self.R, self.intrinsic_matrix, self._distortion_model.coeffs, self._image_shape
            )
            return self.copy(
                rot_world_to_cam=new_R,
                intrinsic_matrix=new_K,
                distortion_model=BrownConradyEx(new_coeffs),
            )
        else:
            # Fisheye or no distortion: just flip R and K
            new_R = self.R.copy()
            new_R[0] *= -1
            new_K = self.intrinsic_matrix.copy()
            new_K[0, 2] = (self._image_shape[1] - 1) - new_K[0, 2]
            new_K[0, 1] *= -1
            return self.copy(rot_world_to_cam=new_R, intrinsic_matrix=new_K)

    def image_rotated(self, angle, anchor=None) -> "Camera":
        """Return a camera adjusted for a rotated image.

        Args:
            angle: Rotation angle in radians (counter-clockwise)
            anchor: Rotation center (x, y). If None, uses image center (requires image_shape).

        Returns:
            New Camera for the rotated image
        """
        if anchor is None:
            if self._image_shape is None:
                raise ValueError("image_shape must be set when anchor is None")
            anchor = (np.array([self._image_shape[1], self._image_shape[0]], np.float32) - 1) / 2
        else:
            anchor = np.asarray(anchor, dtype=np.float32)

        if isinstance(self._distortion_model, BrownConradyEx):
            new_R, new_K, new_coeffs = BrownConradyEx.transform_for_rotation(
                self.R, self.intrinsic_matrix, self._distortion_model.coeffs, angle, anchor
            )
            return self.copy(
                rot_world_to_cam=new_R,
                intrinsic_matrix=new_K,
                distortion_model=BrownConradyEx(new_coeffs),
            )
        else:
            # Fisheye or no distortion: simple rotation
            angle = np.float32(angle)
            sin = np.sin(angle)
            cos = np.cos(angle)
            R_2d = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)

            new_K = self.intrinsic_matrix.copy()
            x = R_2d[1, :] @ new_K[:2, :2]
            x /= np.linalg.norm(x)
            R_ = np.array([[x[1], -x[0]], x], dtype=np.float32)

            new_K[:2, :2] = R_2d @ new_K[:2, :2] @ R_.T
            new_K[:2, 2] = R_2d @ (new_K[:2, 2] - anchor) + anchor

            new_R = self.R.copy()
            new_R[:2] = R_ @ new_R[:2]

            return self.copy(rot_world_to_cam=new_R, intrinsic_matrix=new_K)

    def image_rot90(self, k=1) -> "Camera":
        """Return a camera adjusted for a 90-degree rotated image.

        Requires image_shape to be set.

        Args:
            k: Number of 90-degree rotations (counter-clockwise)

        Returns:
            New Camera for the rotated image
        """
        if self._image_shape is None:
            raise ValueError("image_shape must be set for image_rot90")

        k = k % 4
        if k == 0:
            return self.copy()
        elif k == 1:
            a = (self._image_shape[0] - 1) / 2
            new_shape = (self._image_shape[1], self._image_shape[0])
            return self.image_rotated(np.pi / 2, anchor=(a, a)).copy(image_shape=new_shape)
        elif k == 2:
            return self.image_rotated(np.pi)
        else:  # k == 3
            a = (self._image_shape[1] - 1) / 2
            new_shape = (self._image_shape[1], self._image_shape[0])
            return self.image_rotated(-np.pi / 2, anchor=(a, a)).copy(image_shape=new_shape)

    def undistorted(self, square_pixels=False) -> "Camera":
        """Return a new camera with distortion removed.

        Requires image_shape to be set for proper handling.

        Args:
            square_pixels: If True, adjust intrinsics for square pixels

        Returns:
            New Camera without distortion
        """
        if self._image_shape is None:
            raise ValueError("image_shape must be set to create undistorted camera")

        new_K = self.intrinsic_matrix.copy()
        if square_pixels:
            fx, fy = new_K[0, 0], new_K[1, 1]
            fmean = 0.5 * (fx + fy)
            multiplier = np.array(
                [[fmean / fx, 0, 0], [0, fmean / fy, 0], [0, 0, 1]], np.float32
            )
            new_K = multiplier @ new_K

        return self.copy(intrinsic_matrix=new_K, distortion_model=None)

    def principal_point_centered(self) -> "Camera":
        """Return a new camera with principal point at image center.

        Requires image_shape to be set.

        Returns:
            New Camera with centered principal point
        """
        if self._image_shape is None:
            raise ValueError("image_shape must be set for principal_point_centered")

        new_K = self.intrinsic_matrix.copy()
        new_K[:2, 2] = np.float32([self._image_shape[1] - 1, self._image_shape[0] - 1]) / 2

        return self.copy(intrinsic_matrix=new_K)

    @staticmethod
    def from_fov(fov_degrees, imshape, world_up=(0, -1, 0), side='max'):
        """Create a camera with a given field of view, with centered principal point.

        Args:
            fov_degrees: the field of view along the larger side of the image, in degrees.
            imshape: height and width of the image, for determining the principal point.
            world_up: a world vector that is designated as "pointing up".
        """
        intrinsics = intrinsics_from_fov(fov_degrees, imshape, side)
        return Camera(
            intrinsic_matrix=intrinsics,
            world_up=world_up,
            image_shape=(imshape[0], imshape[1]),
        )

    @staticmethod
    def create2D(imshape=(0, 0)):
        """Create a camera for expressing 2D transformations by using intrinsics only.

        Args:
            imshape: height and width, the principal point of the intrinsics is set at the middle
                of this image size.

        Returns:
            The new camera.
        """
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[:2, 2] = [(imshape[1] - 1) / 2, (imshape[0] - 1) / 2]
        img_shape = (imshape[0], imshape[1]) if imshape != (0, 0) else None
        return Camera(intrinsic_matrix=intrinsics, image_shape=img_shape)

    @property
    def optical_center(self):
        """The optical center (position) of the camera."""
        return self.t


def intrinsics_from_fov(fov_degrees, imshape, side='max'):
    """Create an intrinsic matrix from a field of view and image shape.

    Args:
        fov_degrees: the field of view along the larger side of the image, in degrees.
        imshape: height and width of the image, for determining the principal point.

    Returns:
        The intrinsic matrix.
    """

    if side == 'max':
        sidelength = np.max(imshape[:2])
    elif side == 'min':
        sidelength = np.min(imshape[:2])
    elif side == 'height':
        sidelength = imshape[0]
    elif side == 'width':
        sidelength = imshape[1]
    else:
        raise ValueError(f"Unknown side '{side}' for fov calculation.")

    f = sidelength / (np.tan(np.deg2rad(fov_degrees) / 2) * 2)
    intrinsics = np.array(
        [[f, 0, (imshape[1] - 1) / 2], [0, f, (imshape[0] - 1) / 2], [0, 0, 1]], np.float32
    )
    return intrinsics


def to_homogeneous(points):
    return cv2.convertPointsToHomogeneous(points).squeeze(1)


def from_homogeneous(points):
    return cv2.convertPointsFromHomogeneous(points).squeeze(1)


def visible_subbox(old_camera, new_camera, old_imshape, new_box):
    """Compute the sub-box of `new_box` that would contain valid pixels when reprojecting.

    In other words, compute the part of `new_box` (from `new_camera`'s POV) that has pixels
    that are visible in `old_camera`'s  image of shape `old_imshape`.

    Args:
        old_camera: the original camera
        new_camera: the new camera
        old_imshape: shape of the image of the old camera (height, width)
        new_box: box in the new camera image (x, y, w, h)

    Returns:
        The sub-box of `new_box` (x, y, w, h) as described.
    """
    valid_poly = validity.get_valid_poly_reproj(
        old_camera, new_camera, old_imshape[:2], None
    )
    s_box = shapely.Polygon.from_bounds(*new_box[:2], *(new_box[:2] + new_box[2:]))
    minx, miny, maxx, maxy = valid_poly.intersection(s_box).bounds
    return np.array([minx, miny, maxx - minx, maxy - miny], np.float32)
