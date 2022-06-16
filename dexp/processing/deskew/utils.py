import math

import numpy as np

from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.utils import xpArray
from dexp.utils.backends import Backend


def skew(image: xpArray, psf: xpArray, shift: int, angle: float, zoom: int, axis: int = 0) -> xpArray:
    """Translates, skews and blurs the input image.

    Parameters
    ----------
    image : xpArray
        Input n-dim array.
    psf : xpArray
        Input PSF, must match image dimensions.
    shift : int
        Translation shift.
    angle : float
        Skew angle.
    zoom : int
        Zoom factor.
    axis : int, optional
        Axis to apply the translation and skewing, by default 0

    Returns
    -------
    xpArray
        Skewed image.
    """
    xp = Backend.get_xp_module(image)
    sp = Backend.get_sp_module(image)

    assert image.ndim == psf.ndim

    # Pad:
    pad_width = () + ((0, 0),) * (image.ndim - 1)
    pad_width = [(0, 0) for _ in range(image.ndim)]
    pad_width[axis] = (int(shift * zoom * image.shape[axis] // 2), int(shift * zoom * image.shape[axis] // 2))

    image = np.pad(image, pad_width=pad_width)

    # Add blur:
    psf = psf.astype(dtype=image.dtype, copy=False)
    image = fft_convolve(image, psf)

    # apply skew:
    angle = 45
    matrix = xp.eye(image.ndim)
    matrix[axis, 0] = math.cos(angle * math.pi / 180)
    matrix[axis, 1] = math.sin(angle * math.pi / 180)
    offset = np.zeros(image.ndim)
    offset[axis] = shift

    # matrix = xp.linalg.inv(matrix)
    skewed = sp.ndimage.affine_transform(image, matrix, offset=offset)

    # Add noise and clip
    # skewed += xp.random.uniform(-1, 1)
    skewed = np.clip(skewed, a_min=0, a_max=None)

    # cast to uint16:
    skewed = skewed.astype(dtype=image.dtype)

    return skewed
