Valid Distortion Regions
========================

Lens distortion models are simple functions fitted to calibration data. Within the
calibration region, they accurately describe how light rays bend through the
lens, but outside that region, they can behave erratically. This
page explains how we find the boundary of the valid region and why this matters.

The problem
-----------

Consider the Brown-Conrady distortion model. It maps undistorted normalized
coordinates (x, y) to distorted coordinates (x', y') using a polynomial in
the radial distance r = sqrt(x² + y²). The polynomial was fitted to
calibration points, typically covering the image sensor area.

Outside the calibration region, the polynomial is extrapolating, and polynomials
are notoriously bad at extrapolation. A sixth-degree polynomial that fits
nicely within [-1, 1] can shoot off to infinity or oscillate wildly just
outside that range.

For lens distortion, the failure mode is that the mapping can fold back on
itself. Multiple undistorted points map to the same distorted point. The
distortion is no longer one-to-one, and undistortion becomes ambiguous.

The Jacobian criterion
----------------------

We can detect where the mapping fails by examining the Jacobian matrix. The
Jacobian of the distortion function is the 2x2 matrix of partial derivatives::

    J = | ∂x'/∂x  ∂x'/∂y |
        | ∂y'/∂x  ∂y'/∂y |

The determinant of this matrix tells us how areas transform. If det(J) > 0,
a small region around the point maps to a region with the same orientation.
If det(J) < 0, the orientation flips. If det(J) = 0, the region collapses to
a line or point.

The boundary of the valid region is where det(J) = 0. This is where the
mapping transitions from one-to-one to folding back on itself.

Finding the boundary
--------------------

For the Brown-Conrady model, the Jacobian determinant is a complicated function
of (x, y) and the distortion coefficients (up to 14 with tilt). We could search for zeros in 2D,
but there is a simpler approach: search in polar coordinates.

The distortion is approximately radially symmetric (the dominant terms depend
only on r, not on the azimuthal angle α). So the boundary is approximately a
circle centered at the principal point. We search along radial rays: for each
angle α, find the smallest radius r where det(J) = 0.

The algorithm has three phases:

1. **Find the asymptote limit**: The Brown-Conrady model has a rational
   component with denominator 1 + k5·r² + k6·r⁴ + k7·r⁶. This denominator
   can go to zero, creating an asymptote. We solve the cubic (in r²) to find
   where this happens and use it as an upper bound on the search.

2. **Coarse line search**: For each angle in a coarse sampling (24 angles),
   we shoot a ray outward from the origin. We evaluate the Jacobian determinant
   at many points along the ray, with denser sampling near the origin where
   the boundary often lies. We find where the determinant first becomes
   negative.

3. **Newton refinement**: We interpolate the coarse results to a dense
   sampling (128 angles), then refine each radius using Newton's method.
   Newton requires the derivative of the Jacobian determinant with respect
   to r, which we compute analytically.

The Jacobian determinant
~~~~~~~~~~~~~~~~~~~~~~~~

For a point (x, y) = (r·cos(α), r·sin(α)), the Jacobian determinant is::

    det(J) = (∂x'/∂x)(∂y'/∂y) - (∂x'/∂y)(∂y'/∂x)

Expanding this for the full Brown-Conrady model (12 coefficients, or 14 with tilt) yields
a long expression. We used SymPy to derive the formula and
common subexpression elimination to simplify it. The result is still dozens
of intermediate variables, but it evaluates efficiently.

The derivative ∂det(J)/∂r is even messier. It is also derived symbolically
and simplified. Together, these two expressions enable Newton's method to
converge quickly.

Converting to distorted space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The boundary we find is in undistorted normalized coordinates. But we often
need to check whether a distorted point is valid. So we also need the boundary
in distorted space.

We transform the boundary points through the distortion function. But there
is a subtlety: distortion can reorder points around the circle. A sequence
of points at increasing angles in undistorted space may not be at increasing
angles after distortion.

So after transforming, we sort the boundary points by their distorted angle.
We also add wrap-around points at ±2π to ensure interpolation works across
the ±π boundary.

Using the valid region
~~~~~~~~~~~~~~~~~~~~~~

To check if a point is valid, we convert it to polar coordinates (r, α) and
look up the valid radius for that angle. If the point's radius is less than
the boundary radius, it is valid.

The boundary is stored as two arrays: radii and angles. We use linear
interpolation to find the boundary radius for any angle. This is O(log n)
with binary search, or O(1) if we precompute a lookup table.

Fisheye distortion
------------------

The fisheye (Kannala-Brandt) model is simpler. It maps the incidence angle θ
(angle between the incoming ray and the optical axis) to a radial distance in
the image::

    r_distorted = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹

This is a polynomial in θ, not in the image radius. The undistorted normalized
radius is r_undistorted = tan(θ).

For the mapping to be valid, it must be monotonic: larger θ should give larger
r_distorted. The boundary is where the derivative becomes zero::

    dr_distorted/dθ = 1 + 3k1·θ² + 5k2·θ⁴ + 7k3·θ⁶ + 9k4·θ⁸ = 0

This is a polynomial in θ². We search for the smallest positive root using
line search followed by Newton refinement.

Because the fisheye model is radially symmetric (no dependence on the azimuthal
angle α), the valid region is a circle. We only need to find one radius, not a
boundary curve for each angle.

Why this matters
----------------

Without validity checking, lens undistortion can produce artifacts in the image. Points
outside the valid region may undistort to wrong locations, typically resulting in a reflection effect across the validity boundary, duplicating some parts of the image, or the
iterative undistortion algorithm may fail to converge.

For camera reprojection (warping an image from one camera view to another),
we need to know which parts of the output image have valid data. The valid
region is the intersection of:

- The valid distortion region of the source camera (after undistorting)
- The valid distortion region of the target camera (before distorting)
- The image boundaries of both cameras
- The region visible from both camera orientations (clipping at z=0)

This intersection can be explicitly computed using polygon operations, but we can also obtain a validity mask during the reprojection, by setting and preserving nan values for invalid positions. The result tells
us which pixels in the output have meaningful values and which are undefined.
