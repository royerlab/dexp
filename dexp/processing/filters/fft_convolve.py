import numpy
import scipy.fftpack

from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def fft_convolve(image1: xpArray, image2: xpArray, mode: str = "reflect", in_place: bool = True, internal_dtype=None):
    """
    Fast FFT based convolution.

    Parameters
    ----------
    image1 : First image
    image2 : Second image
    mode : Not supported!
    in_place : If true then the two input images might be modified and reused for the result.

    Returns
    -------
    Convolved image: image1 ○ image2
    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if image1.ndim == image2.ndim == 0:  # scalar inputs
        return image1 * image2
    elif not image1.ndim == image2.ndim:
        raise ValueError("Dimensions do not match.")
    elif image1.size == 0 or image2.size == 0:  # empty arrays
        return xp.asarray([])
    elif image1.dtype != image2.dtype:
        raise ValueError("Two images must have same dtype!")

    if internal_dtype is None:
        internal_dtype = image1.dtype

    if isinstance(Backend.current(), NumpyBackend):
        internal_dtype = numpy.float32

    original_dtype = image1.dtype
    image1 = Backend.to_backend(image1, dtype=internal_dtype, force_copy=False)
    image2 = Backend.to_backend(image2, dtype=internal_dtype, force_copy=False)

    if mode != "wrap":
        pad_width = tuple(tuple((s // 2, s // 2)) for s in image2.shape)
        image1 = xp.pad(image1, pad_width=pad_width, mode=mode)

    s1 = numpy.asarray(image1.shape)
    s2 = numpy.asarray(image2.shape)

    shape = tuple(s1 + s2 - 1)

    fsize = tuple(scipy.fftpack.next_fast_len(x) for x in tuple(shape))

    image1_fft = sp.fft.rfftn(image1, fsize, overwrite_x=in_place)
    image2_fft = sp.fft.rfftn(image2, fsize, overwrite_x=in_place)
    image1_fft *= image2_fft
    del image2_fft
    result = sp.fft.irfftn(image1_fft, overwrite_x=in_place)
    if not in_place:
        del image1_fft

    fslice = tuple(slice(0, int(sz)) for sz in shape)
    result = result[fslice]

    newshape = numpy.asarray(image1.shape)
    currshape = numpy.array(result.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    result = result[tuple(myslice)]

    if mode != "wrap":
        slicing = tuple(slice(s // 2, -(s // 2)) for s in image2.shape)
        result = result[slicing]

    result = result.astype(dtype=original_dtype, copy=False)

    return result
