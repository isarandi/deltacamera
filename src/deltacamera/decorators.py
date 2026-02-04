import functools
import warnings

import numpy as np


class DeprecatingArray(np.ndarray):
    """An ndarray subclass that emits a DeprecationWarning when modified.

    Used for Camera arrays to provide a soft migration path from mutable to immutable API.
    Mutations still work but emit warnings, unless _suppress_warning is True.

    Usage:
        arr = np.asanyarray(data, dtype=np.float32).view(DeprecatingArray)
    """

    def __array_finalize__(self, obj):
        self._suppress_warning = getattr(obj, "_suppress_warning", False)

    def __setitem__(self, key, value):
        if not self._suppress_warning:
            warnings.warn(
                "Direct mutation of Camera arrays is deprecated. "
                "Use immutable methods like zoomed(), rotated(), image_shifted() instead, "
                "or copy() with keyword arguments.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__setitem__(key, value)

    def __reduce__(self):
        # Pickle as regular ndarray - Camera.__setstate__ will re-wrap after unpickling
        return (
            _reconstruct_array,
            (np.asarray(self).tobytes(), self.dtype.str, self.shape),
        )


def _reconstruct_array(data_bytes, dtype_str, shape):
    """Reconstruct a numpy array from pickled bytes."""
    return np.frombuffer(data_bytes, dtype=dtype_str).reshape(shape).copy()


def _suppress_warnings(arr):
    """Suppress deprecation warnings on this array (for legacy mutable API)."""
    if isinstance(arr, DeprecatingArray):
        arr._suppress_warning = True


def _restore_warnings(arr):
    """Restore deprecation warnings on this array."""
    if isinstance(arr, DeprecatingArray):
        arr._suppress_warning = False


def point_transform(f):
    """Decorator to make a function, which transforms multiple points, also accept a single point,
    as well as lists, tuples etc. that can be converted by np.asarray."""

    @functools.wraps(f)
    def wrapped(self, points, *args, **kwargs):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 2:
            return f(self, points, *args, **kwargs)

        reshaped = np.reshape(points, [-1, points.shape[-1]])
        reshaped_result = f(self, reshaped, *args, **kwargs)
        return np.reshape(reshaped_result, [*points.shape[:-1], reshaped_result.shape[-1]])

    return wrapped


def camera_transform(f):
    """Decorator for camera transformation methods.

    Handles deprecation by temporarily suppressing warnings during mutation
    (for backwards compatibility with legacy mutable API).
    Supports `inplace=True` (default, mutates self) and `inplace=False` (returns modified copy).
    """

    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        inplace = kwargs.pop("inplace", True)

        if inplace:
            # Temporarily suppress warnings for legacy mutable API
            _suppress_warnings(self.intrinsic_matrix)
            _suppress_warnings(self.R)
            _suppress_warnings(self.t)
            _suppress_warnings(self.world_up)

            try:
                f(self, *args, **kwargs)
            finally:
                # Restore warnings regardless of success/failure
                _restore_warnings(self.intrinsic_matrix)
                _restore_warnings(self.R)
                _restore_warnings(self.t)
                _restore_warnings(self.world_up)

            return self
        else:
            # Create deep copy and mutate that
            camcopy = self.copy()

            # Suppress warnings on copy during mutation
            _suppress_warnings(camcopy.intrinsic_matrix)
            _suppress_warnings(camcopy.R)
            _suppress_warnings(camcopy.t)
            _suppress_warnings(camcopy.world_up)

            try:
                f(camcopy, *args, **kwargs)
            finally:
                _restore_warnings(camcopy.intrinsic_matrix)
                _restore_warnings(camcopy.R)
                _restore_warnings(camcopy.t)
                _restore_warnings(camcopy.world_up)

            return camcopy

    return wrapped