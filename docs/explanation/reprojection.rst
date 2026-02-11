Camera Reprojection
===================

DeltaCamera is all about changes in camera parameters and how that affects the projection of 3D points to 2D images.
One of the most common operations is reprojection: transforming points or images
from one camera view to another, for example to virtually rotate the camera towards a specific object or to undistort a fisheye image.

This page explains how the reprojection pipeline works in DeltaCamera, the coordinate spaces
involved, and the optimizations that make it practical.

The problem
-----------

We have an image captured by one camera and want to see what it would look like
from a different camera. The two cameras have the same optical center (3D
position), but may differ in:

- **Intrinsics**: focal length, principal point, skew
- **Orientation**: rotation relative to the world
- **Distortion**: lens distortion coefficients

If the optical centers differ, we have parallax, and reprojection becomes
ambiguous without depth information. This library assumes coincident optical
centers.

Coordinate spaces
-----------------

A point passes through several coordinate spaces during reprojection:

1. **Pixel coordinates** (p): The (x, y) position in the image, in pixels.

2. **Normalized distorted** (pn): Pixel coordinates with intrinsics removed::

       pn = K⁻¹ @ [px, py, 1]ᵀ

   This is where the principal point is at the origin and focal length is 1.
   But lens distortion is still present.

3. **Normalized undistorted** (pun): The "true" direction of the ray, with
   distortion removed. For a pinhole camera, pn = pun. For a real lens,
   undistortion is a nonlinear operation.

4. **3D direction**: The ray direction in camera coordinates, (pun_x, pun_y, 1).
   Rotation transforms this direction between camera frames.

The full pipeline
-----------------

To reproject a point from new_camera to old_camera::

    p_new  →  pn_new  →  pun_new  →  pun_old  →  pn_old  →  p_old
         K_new⁻¹    undist     R_old @ R_new.T    dist      K_old

Each step:

1. **Undo intrinsics**: Multiply by K_new⁻¹ to get normalized distorted coords.

2. **Undistort**: Remove new_camera's lens distortion. This is iterative for
   Brown-Conrady (fixed-point + Newton), direct for fisheye.

3. **Rotate**: Apply the relative rotation R_old @ R_new.T. This is a
   perspective transform in normalized coordinates.

4. **Distort**: Apply old_camera's lens distortion. This is a direct formula
   (polynomial evaluation).

5. **Apply intrinsics**: Multiply by K_old to get pixel coordinates.

The result tells us: "to get the color at pixel p_new in new_camera, sample
pixel p_old in old_camera."

Maps vs points
--------------

The library provides two implementations of this pipeline:

- **maps_impl**: Generates a dense coordinate map for every pixel in the output.
  Used for image reprojection via cv2.remap.

- **points_impl**: Transforms a sparse set of points. Used for reprojecting
  landmarks, bounding boxes, etc.

The logic is nearly identical, but maps_impl generates a grid of (h, w) points
while points_impl takes arbitrary input points. Both dispatch to the same
underlying functions based on the lens types of the two cameras.

Lens type combinations
----------------------

The code handles all combinations of three lens types:

- **NONE**: No distortion (pinhole model)
- **USUAL**: Brown-Conrady model (up to 14 parameters with tilt)
- **FISH**: Kannala-Brandt fisheye model

This gives 9 combinations. Each has a specialized function that skips
unnecessary steps. For example, if both cameras have no distortion, we can
use a single homography instead of the full pipeline.

Some combinations are symmetric (undistort on one side, distort on the other),
but the implementations differ because undistortion is iterative while
distortion is direct.

Distortion and undistortion
---------------------------

Distorting a point is straightforward: evaluate the polynomial formula.

Undistorting is harder because the distortion formula is not invertible in
closed form. The implementation uses:

1. **Fixed-point iteration** (5 iterations): Start with the distorted point as
   a guess for the undistorted point, then iterate::

       pun_new = (pn - tangential - radial_offset) / radial_scale

   This converges quickly for small distortions.

2. **Newton refinement** (2 iterations): Use the Jacobian to refine the
   estimate. The Jacobian is computed analytically from the distortion formula.

For fisheye, undistortion solves a 9th-degree polynomial in the incidence angle
θ. Newton's method converges quickly because the polynomial is monotonic in
the valid region.

Precomputed maps
----------------

Undistortion is expensive: it requires multiple iterations per point. For dense
image reprojection, this means millions of iterations.

The library offers a precomputation optimization. For Brown-Conrady:

1. Build a dense grid of normalized distorted points
2. Undistort all of them, storing both the result and the Jacobian
3. For new queries, look up the nearest grid point and use the Jacobian for
   linear interpolation

This trades memory for speed. The precomputed map is cached by distortion
coefficients, so it can be reused across multiple reprojections.

For fisheye, the precomputation is simpler. The distortion is radially
symmetric, so we only need a 1D lookup table indexed by r². This maps
r_distorted² to a scaling factor that converts to r_undistorted.

Validity masking
----------------

Not every output pixel has a valid source. A pixel may be invalid because:

- It falls outside the valid distortion region (see distortion-validity.rst)
- It maps to a point behind the camera (z ≤ 0)
- It maps outside the source image boundaries

The reprojection functions return both the transformed result and a validity
mask. For images, invalid pixels are filled with a border value. For points,
invalid results are NaN.

The validity mask is computed as an RLE mask, which is efficient for the
typical case where the valid region is a single connected blob.

Fast paths
----------

When possible, the code uses OpenCV's fast warp functions:

- **No distortion on either camera**: Use cv2.warpPerspective with a 3x3
  homography.

- **Same rotation and distortion**: Only intrinsics differ. Use cv2.warpAffine
  with a 2x3 affine matrix.

These paths avoid the full pipeline and are significantly faster.

Caching
-------

Two levels of caching are available:

1. **Map caching**: Cache the entire coordinate map for a specific
   (old_camera, new_camera) pair. Useful when reprojecting many images with
   the same camera configuration.

2. **Undistortion map caching**: Cache the precomputed undistortion lookup
   table for a specific set of distortion coefficients. Useful when the
   distortion coefficients are shared across multiple camera configurations.

The caches use LRU eviction and are keyed by serialized camera parameters.
To enable correct cache hits when only the relative rotation matters, the
code transforms to a canonical frame where old_camera.R = I before caching.

sRGB handling
-------------

For high-quality image reprojection, linear interpolation should happen in
linear light space, not in gamma-encoded sRGB. The library provides an option
to decode sRGB to linear 16-bit, interpolate, then encode back to sRGB.

This uses precomputed lookup tables for the sRGB transfer function, making
the conversion fast.

Antialiasing
------------

The basic reprojection ignores aliasing: it samples the source image at a
single point per output pixel. For minification (output pixels larger than
source pixels), this causes aliasing artifacts.

The library offers a simple antialiasing option: render at a higher resolution
(antialias_factor times larger) and downsample with area interpolation. This
is not as sophisticated as a proper reconstruction filter, but it reduces
the worst aliasing artifacts.
