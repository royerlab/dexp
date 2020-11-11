import numpy

from dexp.processing.backends.backend import Backend


def gaussian_kernel_2d(backend: Backend, size: int = 5, sigma: float = 1.0, dtype=numpy.float16):
    """
    Computes a
    Parameters
    ----------
    backend : Backend to use for computation
    size : size in pixels
    sigma

    Returns
    -------

    """
    xp = backend.get_xp_module()
    ax = xp.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = xp.meshgrid(ax, ax)
    kernel = xp.exp(-0.5 * (xp.square(xx) + xp.square(yy)) / xp.square(sigma))
    kernel /= xp.sum(kernel)
    kernel = kernel.astype(dtype=dtype)
    return kernel
