Distortion and In-Plane Image Rotation
=======================================

When we rotate an image in software, and we care about the camera calibration as metadata, we also have to transform the calibration data in accordance with the image rotation. So we want a new camera such that projecting through the
new camera gives the same result as projecting through the original camera and
then rotating the resulting pixel coordinates.

This page derives the transformation rules for distortion coefficients.

Problem formulation
-------------------

We have a camera with intrinsic matrix :math:`K`, rotation :math:`R`, and
distortion coefficients :math:`d`. We rotate the image by angle :math:`\theta` (in-plane).
We want new parameters :math:`K'`, :math:`R'`, :math:`d'` such that for any
world point :math:`p`:

.. math::

   \text{rotate\_pixel}(\text{camera_old.world\_to\_image}(p)) = \text{camera_new.world\_to\_image}(p)

In other words: "project then transform the image" equals "transform the camera then project".

The Brown-Conrady distortion formula
------------------------------------

The 12-parameter Brown-Conrady model maps undistorted normalized coordinates
:math:`(x, y)` to distorted coordinates :math:`(x', y')`. In normalized
coordinates, the principal point is at the origin and the focal length is 1.

The formula:

.. math::

   r^2 &= x^2 + y^2

   a &= \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}

   b &= 2 p_1 y + 2 p_2 x

   x' &= x(a + b) + r^2(p_2 + s_1) + r^4 s_2

   y' &= y(a + b) + r^2(p_1 + s_3) + r^4 s_4

The coefficients fall into three categories:

- **Radial** (:math:`k_1`–:math:`k_6`): The term :math:`a` depends only on
  :math:`r^2`, not on the direction.
- **Tangential** (:math:`p_1`, :math:`p_2`): The term :math:`b` and parts of
  the offset depend on both :math:`x` and :math:`y` separately, making them
  directional.
- **Thin prism** (:math:`s_1`–:math:`s_4`): The offsets :math:`[s_1, s_3]` and
  :math:`[s_2, s_4]` are vectors that point in specific directions.

Radial coefficients are invariant
---------------------------------

Rotation preserves distance from the origin. If we rotate :math:`(x, y)` by
angle :math:`\theta`:

.. math::

   x' &= x \cos\theta - y \sin\theta

   y' &= x \sin\theta + y \cos\theta

Then:

.. math::

   r'^2 &= x'^2 + y'^2 \\
        &= (x \cos\theta - y \sin\theta)^2 + (x \sin\theta + y \cos\theta)^2 \\
        &= x^2 \cos^2\theta - 2xy \cos\theta \sin\theta + y^2 \sin^2\theta \\
        &\quad + x^2 \sin^2\theta + 2xy \sin\theta \cos\theta + y^2 \cos^2\theta \\
        &= x^2(\cos^2\theta + \sin^2\theta) + y^2(\sin^2\theta + \cos^2\theta) \\
        &= x^2 + y^2 = r^2

Since the radial factor :math:`a` depends only on :math:`r^2`, it is unchanged
by rotation. The radial distortion coefficients :math:`k_1`–:math:`k_6` do not
need to be transformed.

Tangential coefficients must rotate
-----------------------------------

The tangential contribution to the scaling factor is:

.. math::

   b = 2 p_1 y + 2 p_2 x

We can write this as a dot product:

.. math::

   b = 2 \begin{bmatrix} p_2 & p_1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
     = 2 \mathbf{p}^\top \mathbf{u}

where :math:`\mathbf{p} = [p_2, p_1]^\top` is the coefficient vector and
:math:`\mathbf{u} = [x, y]^\top` is the point.

Under rotation :math:`R` (which rotates points by angle :math:`\theta`):

.. math::

   \mathbf{u}' &= R \mathbf{u}

   b' &= 2 \mathbf{p}^\top \mathbf{u}' = 2 \mathbf{p}^\top R \mathbf{u}

For the new coefficient vector :math:`\mathbf{p}'` to give the same result with
the rotated point, we need:

.. math::

   \mathbf{p}'^\top \mathbf{u}' &= \mathbf{p}^\top \mathbf{u}

   \mathbf{p}'^\top R \mathbf{u} &= \mathbf{p}^\top \mathbf{u}

This must hold for all :math:`\mathbf{u}`, so:

.. math::

   \mathbf{p}'^\top R = \mathbf{p}^\top \quad \Rightarrow \quad
   \mathbf{p}' = R^\top \mathbf{p} = R^{-1} \mathbf{p}

The tangential coefficients rotate by :math:`-\theta`, or equivalently, they
rotate *with* the coordinate system rather than with the points.

Geometrically, the tangential distortion has a preferred direction (the
direction where the lens decentering is worst). When we rotate the image,
that physical direction stays fixed relative to the sensor, but the coordinate
axes rotate. So the coefficients, which describe that direction in coordinate
terms, must be updated.

::

    Before rotation:              After rotation by θ:

    y                             y'
    ^  tangential                 ^
    |  direction                  |    tangential
    |     ↗                       |    direction
    |   /                         |       ↗
    +-----> x                     +-------> x'

    p = [p₂, p₁]                  p' = R(-θ)·p

The thin prism coefficients
---------------------------

The thin prism terms add offsets proportional to :math:`r^2` and :math:`r^4`:

.. math::

   \Delta x &= r^2 (s_1 + s_2 r^2)

   \Delta y &= r^2 (s_3 + s_4 r^2)

The vector :math:`[\Delta x, \Delta y]` has a direction determined by
:math:`[s_1 + s_2 r^2, s_3 + s_4 r^2]`. At any fixed :math:`r`, this is a
linear combination of two vectors: :math:`[s_1, s_3]` (the :math:`r^2`
coefficient) and :math:`[s_2, s_4]` (the :math:`r^4` coefficient).

Each of these vectors must rotate with the coordinate system, just like the
tangential coefficients:

.. math::

   \begin{bmatrix} s_1' \\ s_3' \end{bmatrix} &= R(-\theta)
   \begin{bmatrix} s_1 \\ s_3 \end{bmatrix}

   \begin{bmatrix} s_2' \\ s_4' \end{bmatrix} &= R(-\theta)
   \begin{bmatrix} s_2 \\ s_4 \end{bmatrix}

The complication: pixel space vs normalized space
-------------------------------------------------

Everything above assumes we rotate in normalized coordinates. But image
rotation happens in pixel coordinates, and the intrinsic matrix :math:`K`
relates them:

.. math::

   \begin{bmatrix} p_x \\ p_y \\ 1 \end{bmatrix} =
   \begin{bmatrix} f_x & \text{skew} & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
   \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}

If :math:`f_x = f_y` (square pixels, no skew), then :math:`K` is a scalar
multiple of identity in the top-left 2×2 block, and rotation in pixel space
corresponds directly to rotation in normalized space.

But if :math:`f_x \neq f_y`, which is typical, the correspondence breaks down.

::

    Normalized space:             Pixel space:

         y                              py
         ^                              ^
         |   · · ·                      |     . .
         | ·       ·                    |   .     .
         |·    +    ·                   |  .   +   .
         | ·       ·                    |   .     .
         |   · · ·                      |     . .
         +---------> x                  +-----------> px

    A circle of radius r           The same points form an
    in normalized coords           ellipse in pixel coords

Rotating the pixel-space ellipse by 45° does not correspond to rotating the
normalized-space circle by 45°. The rotation angles differ because the
axes scale differently.

Why we cannot just use R_pixel
------------------------------

A naive approach: apply the pixel rotation :math:`R_\text{pixel}` to :math:`K`
and use :math:`R_\text{pixel}` for the normalized rotation too:

.. math::

   K_\text{new} = R_\text{pixel} \cdot K \cdot R_\text{pixel}^\top

But when :math:`f_x \neq f_y`, the resulting :math:`K_\text{new}` has
a non-zero [1,0] element (skew in the wrong place).

Example: :math:`K = \begin{bmatrix} 600 & 0 \\ 0 & 400 \end{bmatrix}`,
:math:`\theta = 45°`.

.. math::

   R_\text{pixel} &= \begin{bmatrix} 0.707 & -0.707 \\ 0.707 & 0.707 \end{bmatrix}

   R_\text{pixel} \cdot K &= \begin{bmatrix} 424.3 & -282.8 \\ 424.3 & 282.8 \end{bmatrix}

   R_\text{pixel} \cdot K \cdot R_\text{pixel}^\top &=
   \begin{bmatrix} 353.5 & 70.7 \\ 70.7 & 353.5 \end{bmatrix}

The [1,0] element is 70.7, not zero. This is problematic:

- Standard camera models assume :math:`K_{1,0} = 0` and skew is only in
  :math:`K_{0,1}`
- OpenCV functions expect this form
- Inversion of :math:`K` is more efficient with :math:`K_{1,0} = 0`

The normalized rotation R_norm
------------------------------

We need to find a rotation :math:`R_\text{norm}` in normalized space such that:

.. math::

   R_\text{pixel} \cdot K_{2 \times 2} = K'_{2 \times 2} \cdot R_\text{norm}

and :math:`K'` remains in standard form (:math:`K'_{1,0} = 0`).

Rearranging:

.. math::

   K' = R_\text{pixel} \cdot K \cdot R_\text{norm}^\top

For :math:`K'_{1,0} = 0`, the [1,0] element of
:math:`R_\text{pixel} \cdot K \cdot R_\text{norm}^\top` must vanish. Expanding:

.. math::

   K'_{1,0} = (R_\text{pixel} \cdot K)_{1,:} \cdot R_\text{norm}^\top_{:,0}
            = (R_\text{pixel} \cdot K)_{1,:} \cdot R_\text{norm}_{0,:}^\top

Let :math:`\mathbf{v} = (R_\text{pixel} \cdot K)_{1,:}`. The condition becomes:

.. math::

   \mathbf{v} \cdot R_\text{norm}_{0,:}^\top = 0

So the first row of :math:`R_\text{norm}` must be perpendicular to
:math:`\mathbf{v}`.

Since :math:`R_\text{norm}` is a rotation matrix, its rows are orthonormal.
If the first row is perpendicular to :math:`\mathbf{v}`, the second row must
be parallel to :math:`\mathbf{v}`. The unique choice (up to sign) that
maintains orientation is:

.. math::

   \hat{\mathbf{v}} &= \mathbf{v} / \|\mathbf{v}\|

   R_\text{norm} &= \begin{bmatrix}
   \hat{v}_1 & -\hat{v}_0 \\
   \hat{v}_0 & \hat{v}_1
   \end{bmatrix}

Verification
~~~~~~~~~~~~

The second row of :math:`K'` is:

.. math::

   K'_{1,:} = (R_\text{pixel} \cdot K)_{1,:} \cdot R_\text{norm}^\top
            = \mathbf{v} \cdot R_\text{norm}^\top

.. math::

   K'_{1,0} &= \mathbf{v} \cdot R_\text{norm}_{0,:}^\top
            = \mathbf{v} \cdot [\hat{v}_1, -\hat{v}_0]^\top
            = v_0 \hat{v}_1 - v_1 \hat{v}_0
            = \frac{v_0 v_1 - v_1 v_0}{\|\mathbf{v}\|}
            = 0 \quad \checkmark

   K'_{1,1} &= \mathbf{v} \cdot R_\text{norm}_{1,:}^\top
            = \mathbf{v} \cdot \hat{\mathbf{v}}
            = \|\mathbf{v}\|

So :math:`K'` has zero in position [1,0] as required, and the new
:math:`f_y' = \|\mathbf{v}\|`.

Computing :math:`\mathbf{v}` explicitly:

.. math::

   \mathbf{v} &= [\sin\theta, \cos\theta] \cdot K_{2 \times 2} \\
              &= [\sin\theta \cdot f_x + \cos\theta \cdot 0,\;
                  \sin\theta \cdot \text{skew} + \cos\theta \cdot f_y] \\
              &= [\sin\theta \cdot f_x,\; \sin\theta \cdot \text{skew} + \cos\theta \cdot f_y]

For a camera with no initial skew:

.. math::

   \mathbf{v} = [\sin\theta \cdot f_x,\; \cos\theta \cdot f_y]

The angle of :math:`R_\text{norm}` is:

.. math::

   \theta_\text{norm} = \arctan2(\hat{v}_0, \hat{v}_1)
                      = \arctan2(\sin\theta \cdot f_x,\; \cos\theta \cdot f_y)

When :math:`f_x = f_y`, this simplifies to :math:`\arctan2(\sin\theta, \cos\theta) = \theta`,
so :math:`R_\text{norm} = R_\text{pixel}`. When :math:`f_x \neq f_y`, the angles differ.

Updating the rotation matrix
----------------------------

The camera's rotation matrix :math:`R` (world-to-camera) must also be updated.
The first two rows of :math:`R` describe how world x and y axes map to camera
x and y. After applying :math:`R_\text{norm}` to the normalized coordinates:

.. math::

   R'_{:2} = R_\text{norm} \cdot R_{:2}

This rotates the camera frame in the image plane by the normalized rotation
angle.

The principal point
-------------------

Image rotation happens around an anchor point, typically the image center.
The principal point :math:`(c_x, c_y)` must also be rotated around this anchor:

.. math::

   \begin{bmatrix} c_x' \\ c_y' \end{bmatrix} =
   R_\text{pixel} \cdot \left(
   \begin{bmatrix} c_x \\ c_y \end{bmatrix} - \text{anchor}
   \right) + \text{anchor}

Putting it all together
-----------------------

To rotate a Brown-Conrady camera's image by angle :math:`\theta` around an
anchor point:

1. Compute the pixel rotation matrix :math:`R_\text{pixel}` from :math:`\theta`

2. Compute :math:`\mathbf{v} = R_\text{pixel}[1,:] \cdot K_{2 \times 2}` and
   normalize to :math:`\hat{\mathbf{v}}`

3. Construct :math:`R_\text{norm}` with :math:`\hat{\mathbf{v}}` as its second row

4. Update intrinsics:
   :math:`K'_{2 \times 2} = R_\text{pixel} \cdot K_{2 \times 2} \cdot R_\text{norm}^\top`

5. Update principal point:
   :math:`K'_{:2,2} = R_\text{pixel} \cdot (K_{:2,2} - \text{anchor}) + \text{anchor}`

6. Update world-to-camera rotation: :math:`R'_{:2} = R_\text{norm} \cdot R_{:2}`

7. Rotate tangential coefficients:
   :math:`[p_2', p_1']^\top = R_\text{norm} \cdot [p_2, p_1]^\top`

8. Rotate thin prism coefficients:
   :math:`[s_1', s_3']^\top = R_\text{norm} \cdot [s_1, s_3]^\top`,
   :math:`[s_2', s_4']^\top = R_\text{norm} \cdot [s_2, s_4]^\top`

The radial coefficients :math:`k_1`–:math:`k_6` remain unchanged.

The tilt case (14 parameters)
-----------------------------

The 14-parameter model adds :math:`\tau_x` and :math:`\tau_y`, representing
sensor plane tilt. These act as a homography applied after distortion.

When the image is rotated, the tilt direction must also rotate. But the tilt
is specified as Euler angles (rotation about x, then y), not as a rotation
matrix. Composing the image rotation with the tilt requires converting to a
different Euler angle convention.

The transformation reorders the angles: the original XY tilt plus the image
rotation (about z) is re-expressed as ZXY Euler angles. The Z component
becomes the new coefficient rotation angle, and the XY components become the
new :math:`\tau_x` and :math:`\tau_y`.

Additionally, the tilt homography correction must be applied to :math:`K`.
The details are intricate but follow the same principle: find the
transformation that preserves the projection invariant.

Fisheye cameras
---------------

The fisheye (Kannala-Brandt) model is radially symmetric. It has no tangential
or thin prism terms, only radial coefficients :math:`k_1`–:math:`k_4`. Since
radial distortion is rotationally invariant, the distortion coefficients do
not change when the image is rotated.

Only :math:`K` and :math:`R` need to be updated, using the same
:math:`R_\text{norm}` derivation as above.
