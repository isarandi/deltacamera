"""Temporal smoothing of camera sequences."""

import numpy as np
from scipy.ndimage import convolve1d



def trapezoidal_kernel(ramp_frames, flat_frames):
    """Create a trapezoidal smoothing kernel.

    Ramps linearly from 0 to 1 over ramp_frames, holds at 1 for
    flat_frames, then ramps back down.
    """
    ramp_up = np.linspace(0, 1, ramp_frames)
    ramp_down = np.linspace(1, 0, ramp_frames)
    return np.concatenate([ramp_up, np.ones(flat_frames), ramp_down])


def gaussian_kernel(sigma, truncate=4.0):
    """Create a Gaussian smoothing kernel."""
    radius = int(sigma * truncate + 0.5)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def moving_average_1d(signal, kernel, weights=None):
    """Weighted moving average of a signal via convolution.

    Args:
        signal: shape (n,) or (n, d).
        kernel: 1D kernel, shape (k,).
        weights: per-sample weights, shape (n,), or None.
    """
    signal = np.asarray(signal, np.float64)
    kernel = np.asarray(kernel, np.float64)
    if weights is None:
        weights = np.ones(signal.shape[0], np.float64)
    else:
        weights = np.asarray(weights, np.float64)
    if signal.ndim == 1:
        num = convolve1d(signal * weights, kernel, mode='nearest')
        den = convolve1d(weights, kernel, mode='nearest')
        return np.where(den > 1e-12, num / den, signal)
    out = np.empty_like(signal)
    for d in range(signal.shape[1]):
        num = convolve1d(signal[:, d] * weights, kernel, mode='nearest')
        den = convolve1d(weights, kernel, mode='nearest')
        out[:, d] = np.where(den > 1e-12, num / den, signal[:, d])
    return out


def smooth_cameras(cameras, kernel=None, ramp_frames=15, flat_frames=41):
    """Smooth a camera sequence by independently filtering each component.

    Decomposes into position, look-at target point, focal lengths
    (log-space), and roll. Filters each, then reassembles. This avoids
    coupling artifacts between translation and rotation.

    Args:
        cameras: list of Camera instances.
        kernel: 1D smoothing kernel. If None, a trapezoidal kernel is
            created from ramp_frames and flat_frames.
    """
    if kernel is None:
        kernel = trapezoidal_kernel(ramp_frames, flat_frames)
    n = len(cameras)
    positions = np.stack([c.t for c in cameras])
    targets = np.stack([c.t + c.R[2] * 100 for c in cameras])
    focals = np.stack([[c.intrinsic_matrix[0, 0], c.intrinsic_matrix[1, 1]]
                       for c in cameras])
    rolls = np.array([c.get_pitch_roll()[1] for c in cameras])
    sm_pos = moving_average_1d(positions, kernel)
    sm_tgt = moving_average_1d(targets, kernel)
    sm_foc = np.exp(moving_average_1d(np.log(focals), kernel))
    sm_roll = moving_average_1d(rolls, kernel)
    out = []
    for i in range(n):
        cam = cameras[i].copy(optical_center=sm_pos[i].astype(np.float32))
        cam = cam.turned_towards(
            target_world_point=sm_tgt[i].astype(np.float32))
        cam = cam.rotated(roll=-float(sm_roll[i]))
        K = cam.intrinsic_matrix.copy()
        K[0, 0] = float(sm_foc[i, 0])
        K[1, 1] = float(sm_foc[i, 1])
        cam = cam.copy(intrinsic_matrix=K)
        out.append(cam)
    return out
