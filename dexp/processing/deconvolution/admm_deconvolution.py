
from typing import Optional, List

import napari
import numpy
import scipy
from functools import reduce
from itertools import combinations

from dexp.utils import xpArray
from dexp.processing.backends.backend import Backend


def line_derivative_kernel(axis: int, dim: int) -> xpArray:
    shape = [1] * dim
    shape[axis] = 3
    K = numpy.zeros(shape, dtype=numpy.float32)
    slicing = tuple(slice(None) if i == axis else 0 for i in range(dim))
    K[slicing] = (1, -1, 0)
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
        slicing = tuple(slice(None) if i in axes else 0 for i in range(dim))
        D[slicing] = cross
        kernels.append(D)

    return kernels


def derivative_func(array: xpArray, axes: int, transpose: bool) -> xpArray:
    xp = Backend.get_xp_module()

    left_slice = [slice(None) for _ in range(array.ndim)]
    right_slice = [slice(None) for _ in range(array.ndim)]

    if isinstance(axes, int):
        axes = (axes,)

    for axis in axes:
        left_slice[axis] = slice(None, -1)
        right_slice[axis] = slice(1, None)

    if transpose:
        left_slice, right_slice = right_slice, left_slice
    
    left_slice = tuple(left_slice)
    right_slice = tuple(right_slice)

    out = xp.zeros_like(array)
    out[right_slice] = array[left_slice] - array[right_slice]

    return out


def admm_deconvolution(image: xpArray,
                       psf: xpArray,
                       rho: float = 0.1,
                       gamma: float = 0.01,
                       iterations: int = 10,
                       internal_dtype: Optional[numpy.dtype] = None,
                       display: bool = False) -> xpArray:

    """
    Reference from: http://jamesgregson.ca/tag/admm.html
    """

    backend = Backend.current()

    if display:
        viewer = napari.view_image(backend.to_numpy(image), name='original')

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
    pad_width = [(s // 2, s - s // 2) for s in backproj.shape]
    image = xp.pad(image, pad_width, mode='reflect')
    # original_slice = tuple(slice(p, p + s) for s, (p, _) in zip(original_shape, pad_width))
    original_slice = tuple(
        slice(p - 1, p + s - 1) for s, p in zip(original_shape, backproj.shape)
    )  # I didn't understand why the other slicing is wrong \O/

    # compute data shape for faster fft
    fsize = tuple(scipy.fftpack.next_fast_len(x) for x in image.shape)

    # create derivative kernels
    Ds = [
        backend.to_backend(line_derivative_kernel(a, image.ndim))
        for a in range(image.ndim)
    ] + cross_derivative_kernels(image.ndim)

    # convert derivative kernels to freq space
    fDs = [sp.fft.fftn(D, fsize) for D in Ds]
    del Ds

    # convert input to freq space
    fbackproj = sp.fft.fftn(backproj, fsize, overwrite_x=True)
    fimage = sp.fft.fftn(image, fsize)

    # pre-compute auxiliary values following reference
    nume_aux = fbackproj * fimage
    denom = fbackproj * fbackproj.conj() +\
        rho * reduce(xp.add, (fD.conj() * fD for fD in fDs))
    del fDs

    # compute parameters used for finite difference differenciation (fast diff operator)
    Daxes = list(range(image.ndim)) + list(combinations(range(image.ndim), 2))

    zeros = lambda : xp.zeros(fsize, dtype=internal_dtype)

    # allocate buffers
    I = zeros()  # output image

    Zs = [zeros() for _ in Daxes]
    Us = [zeros() for _ in Daxes]

    for _ in range(iterations):
        # iterations according to reference
        # loop operations are done using generators to reduce memory allocation
        V = rho * reduce(xp.add, (
            derivative_func(Z - U, axes, True)
            for axes, Z, U in zip(Daxes, Zs, Us)
        ))

        fV = sp.fft.fftn(V, overwrite_x=True); del V
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
