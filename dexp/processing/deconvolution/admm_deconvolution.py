
from typing import Callable, Optional, Tuple

import napari
import numpy
import scipy

from dexp.utils import xpArray
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.fft_convolve import fft_convolve


def derivative_kernel(axis: int, fshape: Tuple[int, ...]) -> Tuple[xpArray, xpArray]:
    shape = [1, 1, 1]
    shape[axis-1] = 3
    K = numpy.zeros(shape, dtype=numpy.float64)
    K.flat = (1, 0, -1)
    return K, scipy.fft.fftn(K, shape=fshape)


def ADMM_deconvolution(image: xpArray,
                       psf: xpArray,
                       iterations: int = 40,
                       rho: float = 1.0,
                       gamma: float = 0.1,
                       internal_dtype: Optional[numpy.dtype] = None) -> xpArray:
    # from http://jamesgregson.ca/tag/admm.html
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if image.ndim != psf.ndim:
        raise ValueError("The image and PSF must have same number of dimensions!")

    if internal_dtype is None:
        internal_dtype = numpy.float32

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = Backend.to_backend(image, dtype=internal_dtype)
    psf = Backend.to_backend(psf, dtype=internal_dtype)

    backproj = xp.flip(psf)

    shape = backproj.shape + image.shape
    fsize = tuple(scipy.fftpack.next_fast_len(x) for x in tuple(shape))

    Dx, fDx = derivative_kernel(0, fsize)
    Dy, fDy = derivative_kernel(1, fsize)
    Dz, fDz = derivative_kernel(2, fsize)

    I = xp.zeros(fsize)

    # TODO stopped here

    return deconv

    if mode != 'wrap':
        pad_width = tuple((tuple((s // 2, s // 2)) for s in backproj.shape))
        image = xp.pad(image, pad_width=pad_width, mode=mode)

    s1 = numpy.array(image.shape)
    s2 = numpy.array(backproj.shape)

    shape = tuple(s1 + s2 - 1)
    fsize = tuple(scipy.fftpack.next_fast_len(x) for x in tuple(shape))

    image_fft = sp.fft.rfftn(image, fsize)
    backproj_fft = sp.fft.rfftn(backproj, fsize)

    # deconv_fft = (backproj_fft.conjugate() * image_fft + epsilon) / (backproj_fft * backproj_fft + epsilon)
    deconv_fft = (image_fft + epsilon) / (backproj_fft + epsilon)

    viewer = napari.Viewer()
    viewer.add_image(xp.fft.fftshift(image_fft).real.get(), name='image')
    viewer.add_image(xp.fft.fftshift(backproj_fft).real.get(), name='backprojector')
    viewer.add_image(xp.fft.fftshift(deconv_fft).real.get(), name='deconvolved')

    napari.run()

    deconv = sp.fft.irfftn(deconv_fft)

    fslice = tuple([slice(0, int(sz)) for sz in shape])
    deconv = deconv[fslice]

    newshape = numpy.asarray(image.shape)
    currshape = numpy.array(deconv.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    deconv = deconv[tuple(myslice)]

    if mode != 'wrap':
        slicing = tuple(slice(s // 2, -(s // 2)) for s in backproj.shape)
        deconv = deconv[slicing]

    return deconv.astype(dtype=original_dtype, copy=False)