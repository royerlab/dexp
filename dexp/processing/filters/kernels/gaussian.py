from typing import Tuple, Union

import numpy

from dexp.utils.backends import Backend, NumpyBackend


def gaussian_kernel_nd(
    size: Union[int, Tuple[int, ...]] = 5, ndim: int = None, sigma: float = 1.0, dtype=numpy.float16
):
    """
    Computes a nD Gaussian kernel
    Parameters
    ----------
    ndim : number of dimensions
    size : size in pixels
    sigma : Gaussian sigma
    dtype : dtype for kernel

    Returns
    -------
    nD Gaussian kernel

    """
    backend = Backend.current()
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    if type(backend) is NumpyBackend:
        dtype = numpy.float32

    if type(size) is not tuple and ndim is not None:
        size = (size,) * ndim

    kernel = xp.zeros(shape=size, dtype=dtype)
    slicing = tuple(slice(s // 2, s // 2 + 1, None) for s in size)
    kernel[slicing] = 1
    kernel = sp.ndimage.gaussian_filter(kernel, sigma=sigma)

    kernel /= xp.sum(kernel)

    return kernel
