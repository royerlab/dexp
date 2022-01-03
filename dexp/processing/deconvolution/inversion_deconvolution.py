from typing import Optional

import numpy
import scipy

from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def inversion_deconvolution(
    image: xpArray,
    psf: xpArray,
    epsilon: float = 1e-12,
    mode: str = "wrap",
    internal_dtype: Optional[numpy.dtype] = None,
) -> xpArray:
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

    if mode != "wrap":
        pad_width = tuple(tuple((s // 2, s // 2)) for s in backproj.shape)
        image = xp.pad(image, pad_width=pad_width, mode=mode)

    s1 = numpy.array(image.shape)
    s2 = numpy.array(backproj.shape)

    shape = tuple(s1 + s2 - 1)
    fsize = tuple(scipy.fftpack.next_fast_len(x) for x in tuple(shape))

    image_fft = sp.fft.rfftn(image, fsize)
    backproj_fft = sp.fft.rfftn(backproj, fsize)

    # deconv_fft = (backproj_fft.conjugate() * image_fft + epsilon) / (backproj_fft * backproj_fft + epsilon)
    deconv_fft = (image_fft + epsilon) / (backproj_fft + epsilon)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(xp.fft.fftshift(image_fft).real.get(), name="image")
    viewer.add_image(xp.fft.fftshift(backproj_fft).real.get(), name="backprojector")
    viewer.add_image(xp.fft.fftshift(deconv_fft).real.get(), name="deconvolved")

    napari.run()

    deconv = sp.fft.irfftn(deconv_fft)

    fslice = tuple(slice(0, int(sz)) for sz in shape)
    deconv = deconv[fslice]

    newshape = numpy.asarray(image.shape)
    currshape = numpy.array(deconv.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    deconv = deconv[tuple(myslice)]

    if mode != "wrap":
        slicing = tuple(slice(s // 2, -(s // 2)) for s in backproj.shape)
        deconv = deconv[slicing]

    return deconv.astype(dtype=original_dtype, copy=False)
