from typing import Callable, Tuple

import numpy as np

from dexp.processing.utils.element_wise_affine import element_wise_affine
from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def normalise_functions(
    image: xpArray,
    low: float = 0.0,
    high: float = 1.0,
    minmax: Tuple[float, float] = None,
    quantile: float = 0,
    clip: bool = False,
    do_normalise: bool = True,
    in_place: bool = True,
    dtype=None,
) -> Tuple[Callable, Callable]:
    """Returns a pair of functions: the first normalises the given image to the range [low, high],
    the second denormalises back to the original range (and dtype).
    Usefull when determining the normalisation parameters and doing the actual normalisation must be decoupled,
    for example when splitting an image into chunks and sensing computation to the GPU.


    Parameters
    ----------
    image : image to normalise
    low, high : normalisation range
    minmax : min and max values of the image if already known.
    quantile : if quantile>0 then quantile normalisation is used to find the min and max values.
        Value must be within [0,1]
    clip : clip after normalisation/denormalisation
    do_normalise : If False, the returned functions are pass-through identity functions
        -- usefull to turn on and off normalisation while still using the functions themselves.
    in_place : In-place computation is allowed and inputs may be modified
    dtype : dtype to normalise to.

    Returns
    -------

    """

    xp = Backend.get_xp_module()

    if isinstance(Backend.current(), NumpyBackend):
        dtype = np.float32

    elif dtype is None:
        if image.itemsize <= 2:
            dtype = xp.float16
        elif image.itemsize == 4 or image.itemsize == 8:
            dtype = xp.float32
        else:
            raise ValueError(f"Failed to converted type {image.dtype}")

    original_dtype = image.dtype

    if minmax is None:

        min_value, max_value = 0, 0
        if quantile > 0:
            min_value = xp.quantile(image, q=quantile)
            max_value = xp.quantile(image, q=1 - quantile)

        # if we did not use quantiles or we get some weird result, let's fallback to standard min max:
        if min_value >= max_value:
            min_value = xp.min(image)
            max_value = xp.max(image)

    else:
        min_value, max_value = minmax

    # Normalise:
    norm_denom = max_value - min_value
    if norm_denom == 0:
        norm_denom = 1
    norm_alpha = (high - low) / norm_denom
    norm_beta = low - norm_alpha * min_value

    # Ensure correct type:
    norm_alpha = xp.asarray(norm_alpha, dtype=dtype)
    norm_beta = xp.asarray(norm_beta, dtype=dtype)

    def _norm_function(_image):
        if not do_normalise:
            return _image
        _image = Backend.to_backend(_image, dtype=dtype)
        _image = element_wise_affine(_image, norm_alpha, norm_beta, out=_image if in_place else None)
        if clip:
            _image = _image.clip(low, high, out=_image)
        _image = _image.astype(dtype=dtype, copy=False)
        return _image

    # Denormalise:
    denorm_alpha = (max_value - min_value) / (high - low)
    denorm_beta = min_value - denorm_alpha * low

    # Ensure correct type:
    denorm_alpha = xp.asarray(denorm_alpha, dtype=dtype)
    denorm_beta = xp.asarray(denorm_beta, dtype=dtype)

    def _denorm_function(_image):
        if not do_normalise:
            return _image
        _image = Backend.to_backend(_image, dtype=dtype)
        _image = element_wise_affine(_image, denorm_alpha, denorm_beta, out=_image if in_place else None)
        if clip:
            _image = _image.clip(min_value, max_value, out=_image)
        _image = _image.astype(dtype=original_dtype, copy=False)
        return _image

    return _norm_function, _denorm_function
