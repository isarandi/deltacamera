"""Easing functions for camera trajectory interpolation."""

import numpy as np


def smoothstep(t):
    """Hermite interpolation (C1 continuous).

    Maps [0, 1] to [0, 1] with zero first derivatives at both endpoints.
    Values outside [0, 1] are clamped.
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smootherstep(t):
    """Perlin's smootherstep (C2 continuous).

    Maps [0, 1] to [0, 1] with zero first and second derivatives at both
    endpoints. Smoother than smoothstep but slower to leave the endpoints.
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ramp(x, start, length, ease=smoothstep):
    """Eased ramp: 0 before start, transitions over length, 1 after.

    Args:
        x: frame index (scalar or ndarray).
        start: frame at which the ramp begins.
        length: duration of the transition in frames.
        ease: easing function (default: smoothstep).
    """
    return ease(np.clip((x - start) / length, 0.0, 1.0))
