import numpy

from dexp.processing.backends.backend import Backend


def gaussian_kernel_2d(backend: Backend, size: int = 5, sigma: float = 1.0, dtype=numpy.float16):
    """
    Computes a 2D Gaussian kernel
    Parameters
    ----------
    backend : Backend to use for computation
    size : size in pixels
    sigma : Gaussian sigma

    Returns
    -------
    2D Gaussian kernel

    """
    xp = backend.get_xp_module()
    ax = xp.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = xp.meshgrid(ax, ax)
    kernel = xp.exp(-0.5 * (xp.square(xx) + xp.square(yy)) / xp.square(sigma))
    kernel /= xp.sum(kernel)
    kernel = kernel.astype(dtype=dtype)
    return kernel


def gaussian_kernel_nd(backend: Backend, ndim:int = 3, size: int = 5, sigma: float = 1.0, dtype=numpy.float16):
    """
    Computes a nD Gaussian kernel
    Parameters
    ----------
    backend : Backend to use for computation
    ndim : number of dimensions
    size : size in pixels
    sigma : Gaussian sigma

    Returns
    -------
    nD Gaussian kernel

    """
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    kernel = xp.zeros(shape=(size,)*ndim, dtype=dtype)
    kernel[(slice(size // 2, size // 2 + 1, None),) * ndim] = 1
    kernel = sp.ndimage.gaussian_filter(kernel, sigma=sigma)

    return kernel
