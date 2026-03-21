Deltacamera
===========

Camera calibration manipulation and image warping for computer vision.

Deltacamera provides coordinate transformations between world, camera, and image spaces
with support for Brown-Conrady and Kannala-Brandt (fisheye) lens distortion models.

Features
--------

- **Coordinate transformations** between world, camera, and image space with lens distortion
- **Camera manipulation** with intuitive methods: ``zoomed``, ``rotated``, ``image_resized``, ``turned_towards``
- **Accurate distortion inversion** using Newton's method (more accurate than OpenCV)
- **Valid region tracking** after distortion, extending Leotta et al. to full Brown-Conrady and fisheye models
- **Fast image warping** with antialiasing, linear RGB interpolation, and map caching

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install deltacamera

   # Optional: precompile Numba functions (1-2 min, speeds up first use)
   python -m deltacamera.precompile

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from deltacamera import Camera

   # Create a camera from an intrinsic matrix
   cam = Camera(intrinsic_matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]])

   # Transform 3D world points to 2D image coordinates
   image_points = cam.world_to_image([[1, 2, 3], [4, 5, 6]])

   # Manipulate the camera (returns a new camera)
   cam2 = cam.zoomed(2.0)
   cam3 = cam2.rotated(pitch=0.1)

   # Remove lens distortion
   undistorted_cam = cam.undistorted()

Image Reprojection
~~~~~~~~~~~~~~~~~~

Reproject an image from one camera to another (e.g., to undistort or change the
virtual viewing direction):

.. code-block:: python

   import numpy as np
   from deltacamera import Camera, reproject_image

   # A camera with Brown-Conrady distortion
   cam = Camera(
       intrinsic_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
       distortion_coeffs=[-0.3, 0.1, 0, 0, 0],
       image_shape=(480, 640))

   # Undistort: create a pinhole version and warp the image
   cam_undist = cam.undistorted()
   undistorted_image = reproject_image(image, cam, cam_undist)

   # Rotate the virtual camera and reproject
   cam_rotated = cam.turned_towards(target_image_point=[100, 200])
   cam_rotated = cam_rotated.undistorted()
   cam_rotated = cam_rotated.copy(image_shape=(256, 256))
   crop = reproject_image(image, cam, cam_rotated)

Point Reprojection
~~~~~~~~~~~~~~~~~~

Transform 2D points (e.g., keypoints, bounding box corners) between cameras:

.. code-block:: python

   from deltacamera import reproject_image_points

   # Reproject keypoints from the original camera to the undistorted one
   new_points = reproject_image_points(old_points, cam, cam_undist)

.. toctree::
   :maxdepth: 2
   :hidden:

   API Reference <api/deltacamera/index>
