from typing import Tuple, Union

import numpy

from dexp.utils.backends import Backend


def butterworth_kernel(
    shape: Tuple[int, ...],
    cutoffs: Union[float, Tuple[float, ...]],
    cutoffs_in_freq_units=False,
    epsilon: float = 1,
    order: int = 3,
    frequency_domain: bool = False,
    dtype=numpy.float32,
):
    """

    Parameters
    ----------
    shape : filter shape
    cutoffs : Butterworth cutoffs.
    cutoffs_in_freq_units : If True, the cutoffs are specified in frequency units.
        If False, the units are in normalised within [0,1]
    epsilon : maximum cutoff gain
    order : Butterworth filter order
    frequency_domain : True to return the kernel in the frequency domain
    dtype : dtype to return kernel

    Returns
    -------
    Normalised butterworth filter kernel.

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    ndim = len(shape)

    if type(cutoffs) is not tuple:
        cutoffs = (cutoffs,) * ndim

    if ndim == 1:

        (lx,) = shape
        (cx,) = cutoffs

        x = xp.fft.fftfreq(lx) if cutoffs_in_freq_units else xp.linspace(-1, 1, lx)

        # An array with every pixel = radius relative to center
        freq = xp.abs(x / cx)

    elif ndim == 2:

        ly, lx = shape
        cy, cx = cutoffs

        x = xp.fft.fftfreq(lx) if cutoffs_in_freq_units else xp.linspace(-1, 1, lx)
        y = xp.fft.fftfreq(ly) if cutoffs_in_freq_units else xp.linspace(-1, 1, ly)

        # An array with every pixel = radius relative to center
        freq = ((x / cx) ** 2)[xp.newaxis, xp.newaxis, :] + ((y / cy) ** 2)[xp.newaxis, :, xp.newaxis]

    elif ndim == 3:
        lz, ly, lx = shape
        cz, cy, cx = cutoffs

        x = xp.fft.fftfreq(lx) if cutoffs_in_freq_units else xp.linspace(-1, 1, lx)
        y = xp.fft.fftfreq(ly) if cutoffs_in_freq_units else xp.linspace(-1, 1, ly)
        z = xp.fft.fftfreq(lz) if cutoffs_in_freq_units else xp.linspace(-1, 1, lz)

        # An array with every pixel = radius relative to center
        freq = (
            ((x / cx) ** 2)[xp.newaxis, xp.newaxis, :]
            + ((y / cy) ** 2)[xp.newaxis, :, xp.newaxis]
            + ((z / cz) ** 2)[:, xp.newaxis, xp.newaxis]
        )

    kernel_fft = 1 / xp.sqrt(1.0 + (epsilon**2) * (freq**order))

    kernel_fft = xp.squeeze(kernel_fft)

    if not cutoffs_in_freq_units:
        kernel_fft = sp.fft.ifftshift(kernel_fft)

    if frequency_domain:
        return kernel_fft
    else:
        kernel = sp.fft.fftshift(xp.real(sp.fft.ifftn(kernel_fft)))
        kernel = kernel / kernel.sum()
        kernel = kernel.astype(dtype, copy=False)
        return kernel
