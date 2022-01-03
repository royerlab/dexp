from typing import Tuple, Union

import numpy

from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.filters.kernels.butterworth import butterworth_kernel
from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def butterworth_filter(
    image: xpArray,
    shape=None,
    cutoffs: Union[float, Tuple[float, ...]] = None,
    cutoffs_in_freq_units=False,
    epsilon: float = 1,
    order: int = 3,
    mode: str = "reflect",
    use_fft: bool = False,
    internal_dtype=None,
):
    """
    Applies a Butterworth filter to an image.
    The Butterworth filter is a type of signal processing filter designed to have a frequency response
    as flat as possible in the passband. It is also referred to as a maximally flat magnitude filter.
    It was first described in 1930 by the British engineer and physicist Stephen Butterworth in his paper
    entitled "On the Theory of Filter Amplifiers". (source: wikipedia, https://en.wikipedia.org/wiki/Butterworth_filter)

    Parameters
    ----------
    image : image to apply filter to
    shape : filter shape
    cutoffs : Butterworth cutoffs.
    cutoffs_in_freq_units : If True, the cutoffs are specified in frequency units.
        If False, the units are in normalised within [0,1]
    order : Butterworth filter order
    mode : mode for convolution
    use_fft : True to use FFT
    internal_dtype : internal dtype used for computation

    Returns
    -------
    Filtered image.

    """
    sp = Backend.get_sp_module()

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = Backend.to_backend(image, dtype=internal_dtype)

    if shape is None:
        shape = (11,) * image.ndim
    elif type(shape) is int and image.ndim > 1:
        shape = (shape,) * image.ndim

    if cutoffs is None:
        cutoffs = (0.5,) * image.ndim
    elif type(cutoffs) is float and image.ndim > 1:
        cutoffs = (cutoffs,) * image.ndim

    butterworth_filter = butterworth_kernel(
        shape=shape, cutoffs=cutoffs, cutoffs_in_freq_units=cutoffs_in_freq_units, epsilon=epsilon, order=order
    )

    image = Backend.to_backend(image)

    if use_fft:
        filtered_image = fft_convolve(image, butterworth_filter, mode=mode)
    else:
        filtered_image = sp.ndimage.convolve(image, butterworth_filter, mode=mode)

    filtered_image = filtered_image.astype(original_dtype, copy=False)

    return filtered_image
