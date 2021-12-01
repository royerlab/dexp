
from typing import Optional, List

import napari
import numpy
import scipy
from functools import reduce, partial
from itertools import combinations

from dexp.utils import xpArray
from dexp.processing.backends.backend import Backend


def line_derivative_kernel(axis: int, dim: int) -> xpArray:
    shape = [1] * dim
    shape[axis] = 3
    K = numpy.zeros(shape, dtype=numpy.float32)
    K.flat = (1, -1, 0)
    return K


def cross_derivative_kernels(dim: int) -> List[xpArray]:
    xp = Backend.get_xp_module()

    cross = xp.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 0]])

    kernels = []
    for axes in combinations(range(dim), 2):
        shape = numpy.ones(dim, dtype=int)
        shape[list(axes)] = 3
        D = xp.zeros(shape)
        D.flat = cross
        kernels.append(D)

    return kernels


def admm_deconvolution(image: xpArray,
                       psf: xpArray,
                       iterations: int = 10,
                       rho: float = 0.1,
                       gamma: float = 0.01,
                       internal_dtype: Optional[numpy.dtype] = None,
                       display: bool = False) -> xpArray:

    """
    Reference from: http://jamesgregson.ca/tag/admm.html
    """

    if display:
        viewer = napari.view_image(image)

    backend = Backend.current()

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

    # normalizing
    image = image - image.min()
    image = image / xp.quantile(image, 0.995)

    backproj = xp.flip(psf)

    original_shape = image.shape
    pad_width = [(s // 2, s - s // 2) for s in backproj.shape]
    image = xp.pad(image, pad_width, mode='reflect')
    original_slice = tuple(slice(p, p + s) for s, (p, _) in zip(original_shape, pad_width))

    fsize = tuple(scipy.fftpack.next_fast_len(x) for x in image.shape)

    Ds = [
        backend.to_backend(line_derivative_kernel(a, image.ndim))
        for a in range(image.ndim)
    ] + cross_derivative_kernels(image.ndim)

    fDs = [sp.fft.fftn(D, fsize) for D in Ds]

    # output image
    I = xp.zeros(fsize, dtype=internal_dtype)

    Zs = [I.copy() for _ in range(image.ndim)]
    Us = [xp.zeros(fsize, dtype=internal_dtype) for _ in fDs]

    fbackproj = sp.fft.fftn(backproj, fsize, overwrite_x=True)
    fimage = sp.fft.fftn(image, fsize)

    nume_aux = fbackproj * fimage
    denom = fbackproj * fbackproj.conj() +\
        rho * reduce(xp.add, (fD.conj() * fD for fD in fDs))
    del fDs

    conv = partial(sp.ndimage.convolve, mode='nearest')

    for _ in range(iterations):
        V = rho * reduce(xp.add, (
            conv(Z - U, xp.flip(D)) for D, Z, U in zip(Ds, Zs, Us)
        ))

        fV = sp.fft.fftn(V, overwrite_x=True); del V
        I = sp.fft.ifftn((nume_aux + fV) / denom, overwrite_x=True).real

        if display:
            viewer.add_image(I[original_slice])

        tmps = [conv(I, D) for D in Ds]
        Zs = [shrink(tmp + U) for tmp, U in zip(tmps, Us)]

        Us = [U + tmp - Z for tmp, Z, U in zip(tmps, Zs, Us)]

    I = I[original_slice].astype(original_dtype)
    I = xp.clip(I, 0, None)

    if display:
        napari.run()
    
    return I
