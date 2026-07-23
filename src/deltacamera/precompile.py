import itertools

import deltacamera
from . import validity
from .distortion_models import BrownConradyEx, FisheyeKannalaBrandt
import numpy as np


def precompile():
    print("Precompiling deltacamera Numba functions...")
    imshape = (100, 100)
    camera_undist = deltacamera.Camera.from_fov(60, imshape)
    camera_dist = camera_undist.copy(
        distortion_model=BrownConradyEx(
            np.array(
                [-3.36591e-01, 1.59742e-01, 1.26970e-04, -7.22557e-05, -4.61953e-02],
                dtype=np.float32,
            )
        )
    )
    camera_fish = camera_undist.copy(
        distortion_model=FisheyeKannalaBrandt(
            np.array([0.42649496, -0.62898034, 0.8450709, -0.46660793], dtype=np.float32)
        )
    )

    im = np.zeros((*imshape, 3), dtype=np.uint8)
    depth = np.ones(imshape, dtype=np.float32)
    points = np.random.rand(10, 2).astype(np.float32) * 100

    # Slightly rotated targets with modified coefficients of each lens type, so that
    # same-lens-type pairs (the most common real use case) exercise the make_both_*
    # kernels instead of the equal-camera fast paths.
    camera_undist_b = camera_undist.rotated(yaw=0.05)
    camera_dist_b = camera_dist.rotated(yaw=0.05).copy(
        distortion_model=BrownConradyEx(
            camera_dist._distortion_model.coeffs * np.float32(0.9)
        )
    )
    camera_fish_b = camera_fish.rotated(yaw=0.05).copy(
        distortion_model=FisheyeKannalaBrandt(
            camera_fish._distortion_model.coeffs * np.float32(0.9)
        )
    )

    for cam1, cam2 in itertools.product(
        [camera_undist, camera_dist, camera_fish],
        [camera_undist_b, camera_dist_b, camera_fish_b],
    ):
        deltacamera.reproject_image(
            im, cam1, cam2, precomp_undist_maps=True, use_linear_srgb=False
        )
        deltacamera.reproject_image(
            im, cam1, cam2, precomp_undist_maps=False, use_linear_srgb=True
        )
        deltacamera.reproject_image(im, cam1, cam2, antialias_factor=2, use_linear_srgb=True)
        deltacamera.reproject_image_points(points, cam1, cam2)
        deltacamera.reproject_mask(im, cam1, cam2)
        deltacamera.reproject_depth_map(depth, cam1, cam2)
        deltacamera.reproject_depth_map(depth, cam1, cam2, antialias_factor=2)
        deltacamera.reproject_rgbd(im, depth, cam1, cam2)
        for imshape1, imshape2 in itertools.product([imshape, None], repeat=2):
            validity._get_valid_poly_reproj(cam1, cam2, imshape1, imshape2)

    # Same-distortion zoom pairs exercise the affine fast paths
    for camera in [camera_undist, camera_dist, camera_fish]:
        zoomed = camera.zoomed(1.1)
        deltacamera.reproject_image(im, camera, zoomed, use_linear_srgb=False)
        deltacamera.reproject_depth_map(depth, camera, zoomed)

    for camera in [camera_undist, camera_dist, camera_fish]:
        points3d = camera.image_to_world(points, 1)
        camera.image_to_camera(points, None)
        camera.image_to_camera(points, 1)
        camera.image_to_camera(points, np.ones_like(points[:, 0]))
        camera.world_to_image(points3d)
        points3d_cam = camera.world_to_camera(points3d)
        camera.camera_to_image(points3d_cam)
        camera.camera_to_world(points3d_cam)


if __name__ == "__main__":
    precompile()
