import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.element_wise_affine import element_wise_affine


def sobel_magnitude_filter(backend: Backend,
                           image,
                           exponent: int = 2,
                           normalise_input: bool = True,
                           in_place: bool = False,
                           internal_dtype=numpy.float16):
    """
    Computes the Sobel magnitude filter response for a given image.

    The Sobel operator, sometimes called the Sobel–Feldman operator or Sobel filter,
    is used in image processing and computer vision, particularly within edge detection
    algorithms where it creates an image emphasising edges. It is named after Irwin Sobel
    and Gary Feldman, colleagues at the Stanford Artificial Intelligence Laboratory (SAIL).
    Sobel and Feldman presented the idea of an "Isotropic 3x3 Image Gradient Operator" at
    a talk at SAIL in 1968.[1] Technically, it is a discrete differentiation operator,
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
    backend : Backend to use
    image : image to apply filter on
    exponent : Exponent to use for the magnitude (norm) of the gradient, 2 for L2, and 1 for L1...
    normalise_input : True to normalise input image between 0 and 1 before applying filter
    internal_dtype : dtype fro internal computation.

    Returns
    -------
    Filtered image

    """
    xp = backend.get_xp_module(image)
    sp = backend.get_sp_module(image)
    ndim = image.ndim

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = backend.to_backend(image, dtype=internal_dtype, force_copy=normalise_input and not in_place)

    if normalise_input:
        min_value = xp.min(image)
        max_value = xp.max(image)
        alpha = (1 / (max_value - min_value)).astype(internal_dtype)
        image = element_wise_affine(backend, image, alpha, -min_value)

    sobel_image = xp.zeros_like(image)

    for i in range(ndim):
        sobel_one_axis = xp.absolute(sp.ndimage.sobel(image, axis=i))
        sobel_image += sobel_one_axis ** exponent

    if exponent == 1:
        pass
    elif exponent == 2:
        sobel_image = xp.sqrt(sobel_image, out=sobel_image)

    sobel_image = sobel_image.astype(dtype=original_dtype, copy=False)

    return sobel_image
