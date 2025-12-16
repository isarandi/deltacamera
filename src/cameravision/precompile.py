import itertools

import cameravision
import cameravision.validity
import numpy as np


def precompile():
    print("Precompiling cameravision Numba functions...")
    camera_undist = cameravision.Camera.from_fov(60, (100, 100))
    camera_dist = camera_undist.copy()
    camera_dist.distortion_coeffs = np.array(
        [-3.36591e-01, 1.59742e-01, 1.26970e-04, -7.22557e-05, -4.61953e-02], dtype=np.float32
    )

    camera_fish = camera_undist.copy()
    camera_fish.distortion_coeffs = np.array(
        [0.42649496, -0.62898034, 0.8450709, -0.46660793], dtype=np.float32
    )

    imshape = (100, 100)
    im = np.zeros((*imshape, 3), dtype=np.uint8)
    points = np.random.rand(10, 2).astype(np.float32) * 100

    for cam1, cam2 in itertools.permutations([camera_undist, camera_dist, camera_fish], 2):
        cameravision.reproject_image(
            im, cam1, cam2, imshape, precomp_undist_maps=True, use_linear_srgb=False
        )
        cameravision.reproject_image(
            im, cam1, cam2, imshape, precomp_undist_maps=False, use_linear_srgb=True
        )
        cameravision.reproject_image_points(points, cam1, cam2)
        cameravision.reproject_mask(im, cam1, cam2, imshape)
        for imshape1, imshape2 in itertools.product([imshape, None], repeat=2):
            cameravision.validity.get_valid_poly_reproj(
                cam1, cam2, imshape_old=imshape1, imshape_new=imshape2
            )

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
