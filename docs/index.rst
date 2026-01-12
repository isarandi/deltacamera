Deltacamera
===========

Camera calibration manipulation and image warping for computer vision.

Deltacamera provides coordinate transformations between world, camera, and image spaces
with support for Brown-Conrady and Kannala-Brandt (fisheye) lens distortion models.

Features
--------

- **Coordinate transformations** between world, camera, and image space with lens distortion
- **Camera manipulation** with intuitive methods: ``zoom``, ``rotate``, ``scale_output``, ``turn_towards``
- **Accurate distortion inversion** using Newton's method (more accurate than OpenCV)
- **Valid region tracking** after distortion, extending Leotta et al. to full Brown-Conrady and fisheye models
- **Fast image warping** with antialiasing, linear sRGB interpolation, and map caching

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

   # Manipulate the camera (in-place by default)
   cam.zoom(2.0)
   cam.rotate(axis=[0, 1, 0], angle=0.1)

   # Or get a new camera with inplace=False
   undistorted_cam = cam.undistort(inplace=False)

.. toctree::
   :maxdepth: 2
   :hidden:

   api/index
   explanation/index
