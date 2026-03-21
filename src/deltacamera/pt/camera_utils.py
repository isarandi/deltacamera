import numpy as np
import torch


def camera_to_tensors(camera, device=None):
    """Extract camera parameters as torch tensors.

    Args:
        camera: deltacamera.Camera object
        device: torch device (default: cpu)

    Returns:
        dict with keys:
            'K': (3, 3) intrinsic matrix
            'R': (3, 3) rotation matrix (world-to-camera)
            't': (3,) camera position in world coordinates
            'd': (C,) distortion coefficients, or None
            'image_shape': (h, w) or None
            'is_fisheye': bool
    """
    K = torch.from_numpy(np.asarray(camera.intrinsic_matrix, dtype=np.float32))
    R = torch.from_numpy(np.asarray(camera.R, dtype=np.float32))
    t = torch.from_numpy(np.asarray(camera.t, dtype=np.float32))

    if device is not None:
        K = K.to(device)
        R = R.to(device)
        t = t.to(device)

    d = None
    is_fisheye = False
    if camera.has_distortion():
        coeffs = camera._distortion_model.coeffs
        d = torch.from_numpy(np.array(coeffs, dtype=np.float32))
        if device is not None:
            d = d.to(device)
        is_fisheye = camera.has_fisheye_distortion()

    return {
        'K': K,
        'R': R,
        't': t,
        'd': d,
        'image_shape': camera.image_shape,
        'is_fisheye': is_fisheye,
    }


def cameras_to_tensors(cameras, device=None):
    """Extract parameters from multiple cameras as batched tensors.

    Args:
        cameras: list of deltacamera.Camera objects
        device: torch device

    Returns:
        dict with:
            'K': (N, 3, 3), 'R': (N, 3, 3), 't': (N, 3),
            'd': list of (C,) tensors or Nones,
            'image_shapes': list of (h, w) or Nones,
            'is_fisheye': list of bools
    """
    dicts = [camera_to_tensors(c, device=device) for c in cameras]
    return {
        'K': torch.stack([d['K'] for d in dicts]),
        'R': torch.stack([d['R'] for d in dicts]),
        't': torch.stack([d['t'] for d in dicts]),
        'd': [d['d'] for d in dicts],
        'image_shapes': [d['image_shape'] for d in dicts],
        'is_fisheye': [d['is_fisheye'] for d in dicts],
    }
