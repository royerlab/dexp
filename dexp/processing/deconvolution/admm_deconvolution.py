from functools import partial, reduce
from typing import Optional

import numpy
import scipy

from dexp.processing.deconvolution.admm_utils import (
    derivative_axes,
    first_derivative_func,
    first_derivative_kernels,
    second_derivative_func,
    second_derivative_kernels,
)
from dexp.utils import xpArray
from dexp.utils.backends import Backend


def admm_deconvolution(
    image: xpArray,
    psf: xpArray,
    rho: float = 0.1,
    gamma: float = 0.01,
    iterations: int = 10,
    derivative: int = 1,
    internal_dtype: Optional[numpy.dtype] = None,
    display: bool = False,
) -> xpArray:

    """
    Reference from: http://jamesgregson.ca/tag/admm.html
    """
    if derivative == 1:
        derivative_func = first_derivative_func
        derivative_kernels = first_derivative_kernels
    elif derivative == 2:
        derivative_func = second_derivative_func
        derivative_kernels = second_derivative_kernels
    else:
        raise RuntimeError(f"Derivative must be 1 or 2. Found {derivative}.")

    backend = Backend.current()

    if display:
        import napari

        viewer = napari.view_image(backend.to_numpy(image), name="original")

    def shrink(array: xpArray) -> xpArray:
        return xp.sign(array) * xp.clip(xp.abs(array) - gamma / rho, 0.0, None)

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if image.ndim != psf.ndim:
        raise ValueError("The image and PSF must have same number of dimensions!")

    if internal_dtype is None:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = Backend.to_backend(image, dtype=internal_dtype)
    psf = Backend.to_backend(psf, dtype=internal_dtype)

    # inverting psf
    backproj = xp.flip(psf)

    # padding with reflection for a better deconvolution
    original_shape = image.shape
    pad_width = [(s // 2, s // 2) for s in backproj.shape]
    image = xp.pad(image, pad_width, mode="reflect")
    # original_slice = tuple(slice(p, p + s) for s, (p, _) in zip(original_shape, pad_width))
    original_slice = tuple(
        slice(p - 1, p + s - 1) for s, p in zip(original_shape, backproj.shape)
    )  # I don't understand why the other slicing is wrong \O/

    # compute data shape for faster fft
    fsize = tuple(scipy.fftpack.next_fast_len(x) for x in image.shape)

    # create derivative kernels
    Ds = [backend.to_backend(D) for D in derivative_kernels(image.ndim)]

    # convert derivative kernels to freq space
    fDs = [sp.fft.fftn(D, fsize) for D in Ds]
    del Ds

    # convert input to freq space
    fbackproj = sp.fft.fftn(backproj, fsize, overwrite_x=True)
    fimage = sp.fft.fftn(image, fsize)

    # pre-compute auxiliary values following reference
    nume_aux = fbackproj * fimage
    denom = fbackproj * fbackproj.conj() + rho * reduce(xp.add, (fD.conj() * fD for fD in fDs))
    del fDs

    # compute parameters used for finite difference differenciation (fast diff operator)
    Daxes = derivative_axes(image.ndim)

    zeros = partial(xp.zeros, shape=fsize, dtype=internal_dtype)

    # allocate buffers
    I = zeros()  # output image

    Zs = [zeros() for _ in Daxes]
    Us = [zeros() for _ in Daxes]

    for _ in range(iterations):
        # iterations according to reference
        # loop operations are done using generators to reduce memory allocation
        V = rho * reduce(xp.add, (derivative_func(Z - U, axes, True) for axes, Z, U in zip(Daxes, Zs, Us)))

        fV = sp.fft.fftn(V, overwrite_x=True)
        del V
        I = sp.fft.ifftn((nume_aux + fV) / denom, overwrite_x=True).real

        if display:
            viewer.add_image(backend.to_numpy(I[original_slice]))

        tmps = [derivative_func(I, axes, False) for axes in Daxes]
        Zs = [shrink(tmp + U) for tmp, U in zip(tmps, Us)]

        for tmp, Z, U in zip(tmps, Zs, Us):
            U += tmp - Z

    # normalization
    I = I[original_slice].astype(original_dtype)
    I -= I.min()
    I /= I.max()

    if display:
        napari.run()

    return I
