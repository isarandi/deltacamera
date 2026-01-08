import cv2
import numpy as np


def allclose_or_nones(a, b):
    """Check if all corresponding values in arrays a and b are close to each other in the sense of
    np.allclose, or both a and b are None, or one is None and the other is filled with zeros.
    """

    if a is None and b is None:
        return True

    if a is None:
        return not cv2.hasNonZero(b)

    if b is None:
        return not cv2.hasNonZero(a)

    if a.shape != b.shape:
        return False

    return np.allclose(a, b)


def equal_or_nones(a, b):
    """Check if all corresponding values in arrays a and b are close to each other in the sense of
    np.allclose, or both a and b are None, or one is None and the other is filled with zeros.
    """

    if a is None and b is None:
        return True

    if a is None:
        return not cv2.hasNonZero(b)

    if b is None:
        return not cv2.hasNonZero(a)

    return np.array_equal(a, b)


def unit_vec(v):
    return v / np.linalg.norm(v)
