SymPy Code Generation
=====================

Lens distortion models are fitted to calibration data. In some cases, outside the calibration
region, they can fold back on themselves and multiple input (undistorted) points can map to the same
distorted output. To find the boundary where this happens, we look for where the Jacobian
determinant of the distortion function has a zero-crossing from positive to negative. That boundary marks the
edge of the valid region.

Finding this boundary requires evaluating the Jacobian determinant at many
points. We use Newton's method to locate the zeros precisely, which also
requires the derivative of the determinant with respect to radius.

These expressions are complicated: the Brown-Conrady model has 12 coefficients,
rational functions, and trigonometry. Hand-deriving them invites mistakes. We
use SymPy to define the distortion symbolically and derive everything else.
This page explains the workflow and the tricks that make it practical.

The distortion function
-----------------------

The Brown-Conrady distortion maps undistorted normalized coordinates (x, y) to
distorted coordinates (x', y'). The full model has radial, tangential, and
thin-prism components::

    r² = x² + y²

    # Radial (rational)
    a = (1 + k0·r² + k1·r⁴ + k4·r⁶) / (1 + k5·r² + k6·r⁴ + k7·r⁶)

    # Tangential
    b = 2·(k3·x + k2·y)

    # Thin prism
    c1 = (k8 + k3 + k9·r²)·r²
    c2 = (k10 + k2 + k11·r²)·r²

    # Distorted coordinates
    x' = x·(a + b) + c1
    y' = y·(a + b) + c2

To find the valid region boundary, we need to find where the Jacobian determinant
is zero. And to use Newton's method for that search, we need the derivative of
the determinant with respect to r.

The polar coordinate trick
--------------------------

We could directly compute the Jacobian with respect to (x, y), but this turns out to be computationally more difficult for the SymPy symbolic computation library.
Instead, we can make the task simpler by switching to polar coordinates, since the distortion is approximately radially symmetric. Polar coordinates are defined as
(r, θ) where x = r·cos(θ), y = r·sin(θ).

Computing the Jacobian with respect to (r, θ) produces simpler intermediate
expressions. The polar Jacobian relates to the Cartesian one by::

    J_rt = J_xy @ J_polar

where J_polar is the Jacobian of the polar-to-Cartesian transformation::

    J_polar = | cos(θ)   -r·sin(θ) |
              | sin(θ)    r·cos(θ) |

The determinant of J_polar is r. So::

    det(J_rt) = det(J_xy) · r
    det(J_xy) = det(J_rt) / r

We compute det(J_rt) symbolically, divide by r, and let SymPy differentiate
the result with respect to r. SymPy applies the quotient rule automatically.

Common subexpression elimination
--------------------------------

The raw Jacobian determinant expression is thousands of characters long. Many
subexpressions appear multiple times. SymPy's ``cse()`` function extracts them
into intermediate variables::

    from sympy import cse

    common, (det_simp, det_prime_simp) = cse([det_xy, det_xy_prime])

For the 12-parameter model, this produces about 50 intermediate variables. The
final expressions for the determinant and its derivative reference these
variables instead of repeating the same computations.

Simplify after CSE
------------------

CSE only finds literal subexpression matches. It does not do algebraic
simplification. Consider::

    x7 = r*x4 - r*x6

CSE sees ``r*x4`` and ``r*x6`` as separate subexpressions. It does not notice
that we could factor out r::

    x7 = r*(x4 - x6)

This saves a multiplication. We catch these by calling ``.simplify()`` on each
CSE variable after extraction::

    common_simplified = []
    for var, expr in common:
        common_simplified.append((var, expr.simplify()))

The simplification is not free—it takes time—but the resulting code is tighter.

From SymPy to Numba
-------------------

SymPy's code printer produces Python code with ``math.sin``, ``math.cos``, and
power operators like ``x**2``. For Numba, we want ``np.sin``, ``np.cos``, and
explicit multiplications (which are faster than ``pow`` calls)::

    def transform_code(code):
        code = code.replace("math.sin", "np.sin")
        code = code.replace("math.cos", "np.cos")
        code = re.sub(r"(\w+)\*\*2", r"(\1*\1)", code)
        code = re.sub(r"(\w+)\*\*3", r"(\1*\1*\1)", code)
        # etc.
        return code

The generated function is decorated with ``@numba.njit`` and can be called in
tight loops without Python overhead.

Why hand-optimization still wins
--------------------------------

The generated code works, but it is about 20-30% slower than hand-optimized
code. The reasons are instructive.

**Division avoidance.** The generated code computes ``x0 = 1/r`` upfront and
uses it later as ``fval = x0 * x46``. The hand-optimized version restructures
the computation to avoid the division entirely, computing the result as
``fval = x22 * (...) - x36 * (...)``. Divisions are roughly 20x slower than
multiplications.

**Reciprocal precomputation.** When a reciprocal like ``1/(x17*x8 + 1)`` appears
multiple times, the hand version computes it once (``x19``) and reuses it. SymPy
may generate the division in multiple places.

**Loop-invariant hoisting.** The coefficients k0 through k11 are constant across
all calls. Sums like ``k3 + k8`` and ``k10 + k2`` could be precomputed once
outside the function. The hand version does this; the generated version does not.

**Grouping for allocation.** In vectorized code, different ways of grouping
operations affect how many intermediate arrays are allocated. The hand version
was tuned to minimize allocation; the generated version was not.

These optimizations require understanding the structure of the computation, not
just mechanical transformations. For production use, we keep the hand-optimized
versions. The codegen scripts serve as documentation and as a way to verify
correctness.

Verifying correctness
---------------------

The generated code is the ground truth. We trust SymPy's differentiation. The
hand-optimized code must match it numerically.

The test suite compares both versions on many random inputs::

    for r, t, d in test_cases:
        f_gen, df_gen = jacobian_det_and_prime_polar_generated(r, t, d)
        f_hand, df_hand = jacobian_det_and_prime_polar(r, t, d)
        assert np.allclose(f_gen, f_hand)
        assert np.allclose(df_gen, df_hand)

We also verify the derivative against finite differences::

    eps = 1e-5
    df_numerical = (f(r + eps) - f(r - eps)) / (2 * eps)
    assert np.allclose(df_analytical, df_numerical, rtol=1e-4)

This caught a bug in an early hand-optimized version of the 14-parameter model.

The 14-parameter extension
--------------------------

The 14-parameter model adds tilt: two angles (τ_x, τ_y) that model sensor tilt
relative to the optical axis. The tilt applies a rotation followed by perspective
division::

    R = Ry(τ_y) @ Rx(τ_x)
    [x_tilt, y_tilt, z]ᵀ = R @ [x_d, y_d, 1]ᵀ
    x'' = x_tilt / z
    y'' = y_tilt / z

The Jacobian of this perspective division has determinant 1/z³. The full
14-parameter Jacobian determinant combines the 12-parameter distortion Jacobian
with this tilt factor.

