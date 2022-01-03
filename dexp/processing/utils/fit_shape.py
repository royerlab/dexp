from typing import Tuple

from dexp.utils import xpArray
from dexp.utils.backends import Backend


def fit_to_shape(array: xpArray, shape: Tuple[int, ...]) -> xpArray:
    """
    Pads or crops an array to attain a certain shape

    Parameters
    ----------
    backend : backend to use
    array : array to pad
    shape : shape to attain by cropping or padding each dimension

    Returns
    -------
    Array of requested shape

    """

    length_diff = tuple(u - v for u, v in zip(shape, array.shape))

    if any(x < 0 for x in length_diff):
        # we need to crop at least one dimension:
        slicing = tuple(slice(0, s) for s in shape)
        array = array[slicing]

    # Independently of whether we had to crop a dimension, we proceed with eventual padding:
    length_diff = tuple(u - v for u, v in zip(shape, array.shape))

    if any(x > 0 for x in length_diff):
        xp = Backend.get_xp_module()
        pad_width = tuple(tuple((0, d)) for d in length_diff)
        array = xp.pad(array, pad_width=pad_width)

    return array
