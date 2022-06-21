import numpy

from dexp.processing.utils.normalise import Normalise
from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def sobel_filter(
    image: xpArray,
    exponent: int = 2,
    gamma: float = 1,
    log_compression: bool = True,
    normalise_input: bool = True,
    in_place_normalisation: bool = False,
    internal_dtype=None,
):
    """
    Computes the Sobel magnitude filter response for a given image.

    The Sobel operator, sometimes called the Sobel–Feldman operator or Sobel filter,
    is used in image processing and computer vision, particularly within edge detection
    algorithms where it creates an image emphasising edges. It is named after Irwin Sobel
    and Gary Feldman, colleagues at the Stanford Artificial Intelligence Laboratory (SAIL).
    Sobel and Feldman presented the idea of an "Isotropic 3x3 Image Gradient Operator" at
    a talk at SAIL in 1968. Technically, it is a discrete differentiation operator,
    computing an approximation of the gradient of the image intensity function.
    At each point in the image, the result of the Sobel–Feldman operator is either the
    corresponding gradient vector or the norm of this vector. The Sobel–Feldman operator
    is based on convolving the image with a small, separable, and integer-valued filter
    in the horizontal and vertical directions and is therefore relatively inexpensive in
    terms of computations. On the other hand, the gradient approximation that it produces
    is relatively crude, in particular for high-frequency variations in the image.

    (Source: wikipedia, https://en.wikipedia.org/wiki/Sobel_operator)

    Parameters
    ----------
    image : image to apply filter on
    exponent : Exponent to use for the magnitude (norm) of the gradient, 2 for L2, and 1 for L1...
    gamma : After normalisation, applies thte given gamma value.
    log_compression : Before normalisation, applies log compression -- usefull to reduce the impact
        of high intensity values versus contrast.
    normalise_input : True to normalise input image between 0 and 1 before applying filter
    in_place_normalisation : If True then input image can be modified during normalisation.
    internal_dtype : dtype for internal computation.

    Returns
    -------
    Filtered image

    """
    if isinstance(Backend.current(), NumpyBackend):
        internal_dtype = numpy.float32

    xp = Backend.get_xp_module(image)
    sp = Backend.get_sp_module(image)
    ndim = image.ndim

    if internal_dtype is None:
        internal_dtype = image.dtype

    original_dtype = image.dtype
    force_copy = normalise_input and not in_place_normalisation
    image = Backend.to_backend(image, dtype=internal_dtype, force_copy=force_copy)

    if log_compression:
        image = xp.log1p(image, out=image if force_copy else None)

    normalise = Normalise(image, do_normalise=normalise_input, dtype=internal_dtype)

    image = normalise.forward(image)

    if gamma != 1:
        image **= gamma

    sobel_image = xp.zeros_like(image)

    for i in range(ndim):
        sobel_one_axis = xp.absolute(sp.ndimage.sobel(image, axis=i))
        sobel_image += sobel_one_axis**exponent

    if exponent == 1:
        pass
    elif exponent == 2:
        sobel_image = xp.sqrt(sobel_image, out=sobel_image)

    sobel_image = sobel_image.astype(dtype=original_dtype, copy=False)

    return sobel_image
