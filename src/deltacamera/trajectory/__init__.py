"""Camera trajectory primitives for deltacamera.

Build flight trajectories, follow-cam pipelines, and real-time tracking
cameras from composable position paths, orientation policies, easing
functions, and smoothing filters.

Requires: numpy, scipy (both already deltacamera dependencies).
"""

from .easing import smoothstep, smootherstep, ramp
from .paths import lerp, cylindrical_arc, circle_positions
from .orientation import upright_rotation, UprightTracker
from .interpolation import slerp_cameras, interpolate_cameras, DepartureBlend
from .smoothing import smooth_cameras, trapezoidal_kernel, gaussian_kernel, moving_average_1d
from .tracking import TrackingCamera

__all__ = [
    'smoothstep', 'smootherstep', 'ramp',
    'lerp', 'cylindrical_arc', 'circle_positions',
    'upright_rotation', 'UprightTracker',
    'slerp_cameras', 'interpolate_cameras', 'DepartureBlend',
    'smooth_cameras', 'trapezoidal_kernel', 'gaussian_kernel',
    'moving_average_1d',
    'TrackingCamera',
]
