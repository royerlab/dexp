from typing import Tuple

import numpy as np
from scipy.signal._signaltools import _centered

from dexp.utils import xpArray


def fit_to_shape(array: xpArray, shape: Tuple[int], **kwargs) -> xpArray:
    """Pads and crop array to fit shape, keyword arguments are used for the padding function."""
    if array.ndim != len(shape):
        raise ValueError(f"Arrays must have the same number of dimensions. Found {array.ndim} and {len(shape)}")

    shape_dif = shape - np.asarray(array.shape)
    if np.any(shape_dif > 0):
        pad_width = tuple((d // 2, d - d // 2) if d > 0 else (0, 0) for d in shape_dif)
        array = np.pad(array, pad_width, **kwargs)

    array = _centered(array, shape)
    return array
