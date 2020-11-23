from typing import Tuple, Union

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def butterworth_kernel(backend: Backend,
                       shape: Tuple[int, ...],
                       cutoffs: Union[float, Tuple[float, ...]],
                       epsilon: float = 1,
                       order: int = 3,
                       frequency_domain: bool = False,
                       dtype = numpy.float32):
    """

    Parameters
    ----------
    backend : Backend to use
    shape : filter shape
    cutoffs : Cutoffs in normalise k-space.
    epsilon : maximum cutoff gain
    order : order

    Returns
    -------
    Normalised butterworth filter kernel.

    """
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    ndim = len(shape)

    if type(cutoffs) is not tuple:
        cutoffs = (cutoffs,)*ndim

    if ndim == 2:

        ly, lx = shape
        cy, cx = cutoffs

        x = xp.fft.fftfreq(lx)
        y = xp.fft.fftfreq(ly)

        # An array with every pixel = radius relative to center
        freq = xp.sqrt(((x / cx) ** 2)[xp.newaxis, xp.newaxis, :] + ((y / cy) ** 2)[xp.newaxis, :, xp.newaxis])

    elif ndim == 3:
        lz, ly, lx = shape
        cz, cy, cx = cutoffs

        x = xp.fft.fftfreq(lx)
        y = xp.fft.fftfreq(ly)
        z = xp.fft.fftfreq(lz)

        # An array with every pixel = radius relative to center
        freq = xp.sqrt(((x / cx) ** 2)[xp.newaxis, xp.newaxis, :] + ((y / cy) ** 2)[xp.newaxis, :, xp.newaxis] + ((z / cz) ** 2)[:, xp.newaxis, xp.newaxis])

    kernel_fft = 1 / (1.0 + (epsilon*freq) ** (2 * order)) ** 0.5

    kernel_fft = xp.squeeze(kernel_fft)

    if frequency_domain:
        kernel_fft = kernel_fft.astype(dtype, copy=False)
        return kernel_fft
    else:
        kernel = sp.fft.fftshift(xp.real(sp.fft.ifftn(kernel_fft)))
        kernel = kernel / kernel.sum()
        kernel = kernel.astype(dtype, copy=False)
        return kernel

