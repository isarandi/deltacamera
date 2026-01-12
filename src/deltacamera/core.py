from typing import Optional, Sequence, TYPE_CHECKING, Union

import cv2
import numpy as np
import shapely
from scipy.spatial.transform import Rotation

from . import coordframes, distortion
from .decorators import camera_transform, point_transform
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
            self.extrinsic_matrix = np.asarray(extrinsic_matrix, dtype=dtype)
        else:
            self.R = np.asarray(rot_world_to_cam, dtype=dtype)
            if optical_center is not None:
                self.t = np.asarray(optical_center, dtype=dtype)
            else:
                self.t = -self.R.T @ np.asarray(trans_after_rot, dtype=dtype)

        self.intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=dtype)
        if distortion_coeffs is None:
            self.distortion_coeffs = None
        else:
            self.distortion_coeffs = np.asarray(distortion_coeffs, dtype=dtype)
            if not cv2.hasNonZero(self.distortion_coeffs):
                self.distortion_coeffs = None

        self.world_up = np.asarray(world_up, dtype=dtype).copy()
        self.world_up /= np.linalg.norm(self.world_up)

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
                pun, self.distortion_coeffs, check_validity=validate_distortion, dst=pn_dst
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
                pun, self.distortion_coeffs, check_validity=validate_distortion, dst=pun
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
                self.distortion_coeffs,
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
        is_valid = np.logical_and(is_valid, cv2.inRange(
            im_points[np.newaxis], lowerb=(0, 0), upperb=(float(imsize[0]), float(imsize[1]))
        ).squeeze(0))
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
            self.distortion_coeffs[[[3], [2]]] = R_ @ self.distortion_coeffs[[[3], [2]]]
            if self.distortion_coeffs.shape[0] > 8:
                self.distortion_coeffs[[[8, 9], [10, 11]]] = (
                    R_ @ self.distortion_coeffs[[[8, 9], [10, 11]]]
                )

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
        return (
            self.distortion_coeffs is not None
            and self.distortion_coeffs.shape[0] == 4
            and cv2.hasNonZero(self.distortion_coeffs)
        )

    def has_nonfisheye_distortion(self):
        return (
            self.distortion_coeffs is not None
            and self.distortion_coeffs.shape[0] != 4
            and cv2.hasNonZero(self.distortion_coeffs)
        )

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
                    self.distortion_coeffs,
                    imsize,
                    np.eye(3),
                    new_size=new_imsize,
                    balance=alpha_balance,
                )
            else:
                self.intrinsic_matrix = cv2.getOptimalNewCameraMatrix(
                    self.intrinsic_matrix,
                    self.distortion_coeffs,
                    imsize,
                    alpha_balance,
                    new_imsize,
                    centerPrincipalPoint=center_principal_point,
                )[0]

        self.distortion_coeffs = None
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
            cam.distortion_coeffs = None
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
        if self.distortion_coeffs is not None and len(self.distortion_coeffs) > 4:
            self.distortion_coeffs[3] *= -1
            if len(self.distortion_coeffs) > 8:
                self.distortion_coeffs[[8, 9]] *= -1

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
        return np.rad2deg(2 * np.arctan(np.max(imshape[:2] / (2 * focals))))

    def get_distortion_coeffs(
        self, n_coeffs_min: int = 5, n_coeffs_max: Optional[int] = None
    ) -> np.ndarray:
        """Get the distortion coefficients of the camera."""
        if self.distortion_coeffs is None:
            return np.zeros(shape=(n_coeffs_min,), dtype=np.float32)
        elif len(self.distortion_coeffs) == 4:
            # Fisheye exception, this should be handled with a better API with an
            # enum for the distortion model. Currently the only signifier is len==4 for fisheye.
            return self.distortion_coeffs
        elif len(self.distortion_coeffs) < n_coeffs_min:
            return np.pad(self.distortion_coeffs, (0, n_coeffs_min - len(self.distortion_coeffs)))
        else:
            return self.distortion_coeffs[:n_coeffs_max]

    def has_distortion(self) -> bool:
        """Check if the camera has nonzero lens distortion."""
        return self.distortion_coeffs is not None and cv2.hasNonZero(self.distortion_coeffs)

    def allclose(self, other_camera):
        """Check if all parameters of this camera are close to corresponding parameters
        of `other_camera`.

        Args:
            other_camera: the camera to compare to.

        Returns:
            True if all parameters are close, False otherwise.
        """
        return (
            np.allclose(self.intrinsic_matrix, other_camera.intrinsic_matrix)
            and np.allclose(self.R, other_camera.R)
            and np.allclose(self.t, other_camera.t)
            and allclose_or_nones(self.distortion_coeffs, other_camera.distortion_coeffs)
        )

    def is_equal(self, other):
        return (
            np.array_equal(self.intrinsic_matrix, other.intrinsic_matrix)
            and np.array_equal(self.R, other.R)
            and np.array_equal(self.t, other.t)
            and equal_or_nones(self.distortion_coeffs, other.distortion_coeffs)
        )

    # Probably not a good idea. Probably first the class should be made immutable
    # There are tradeoffs in that. The default id based hash may also be fine in that case
    # if we expect that the same camera object is not created multiple times.
    # The current situation is non-ideal because the class is treated as mutable
    # but an lru_cache won't recomute the value if the camera is changed.
    # However, mutating the hash value along with the parameters would also be a bad idea,
    # because dicts will work in unexpected ways. The clean solution is probably immutablitity
    # even with the performance hit or having to copy every parameter at every change.
    # def __hash__(self):
    #     if self.distortion_coeffs is not None:
    #         return hash((
    #             self.intrinsic_matrix.tobytes(),
    #             self.R.tobytes(),
    #             self.t.tobytes(),
    #             self.distortion_coeffs.tobytes()))
    #     else:
    #         return hash((
    #             self.intrinsic_matrix.tobytes(),
    #             self.R.tobytes(),
    #             self.t.tobytes()))

    def copy(self) -> "Camera":
        c = Camera.__new__(Camera)
        c.intrinsic_matrix = self.intrinsic_matrix.copy()
        c.R = self.R.copy()
        c.t = self.t.copy()
        if self.distortion_coeffs is not None:
            c.distortion_coeffs = self.distortion_coeffs.copy()
        else:
            c.distortion_coeffs = None
        c.world_up = self.world_up.copy()
        return c

    @staticmethod
    def from_fov(fov_degrees, imshape, world_up=(0, -1, 0), side='max'):
        """Create a camera with a given field of view, with centered principal point.

        Args:
            fov_degrees: the field of view along the larger side of the image, in degrees.
            imshape: height and width of the image, for determining the principal point.
            world_up: a world vector that is designated as "pointing up".
        """
        intrinsics = intrinsics_from_fov(fov_degrees, imshape, side)
        return Camera(intrinsic_matrix=intrinsics, world_up=world_up)

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
        return Camera(intrinsic_matrix=intrinsics)

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
