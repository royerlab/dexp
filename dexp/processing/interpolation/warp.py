

from typing import Tuple

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.utils.nan_to_zero import nan_to_zero
from dexp.processing.utils.normalise import normalise

def warp(backend: Backend,
         image,
         vector_field,
         internal_dtype=numpy.float16):
    """
    Applies a warp transform (piece wise linear or constant) to an image based on a vector field.

    Parameters
    ----------
    backend : backend to use
    image : image to deconvolve
    vector_field : point-spread-function (must have the same number of dimensions as image!)

    Returns
    -------
    Warped image

    """

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    if image.ndim+1 == vector_field.ndim:
        raise ValueError("Vector field must have one additional dimension")

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = backend.to_backend(image, dtype=internal_dtype)
    vector_field = backend.to_backend(vector_field, dtype=internal_dtype)

    ## something happens here:
    from dexp.processing.backends.cupy_backend import CupyBackend
    if type(backend) is NumpyBackend:
        raise NotImplemented("Warping not yet implemented for the Numpy backend.")
    elif type(backend) is CupyBackend:
        from dexp.processing.interpolation._cupy.warp import _warp_1d_cupy, _warp_2d_cupy, _warp_3d_cupy
        params = (backend, image, vector_field, internal_dtype)
        if image.ndim == 1:
            result = _warp_1d_cupy(*params)
        if image.ndim == 2:
            result = _warp_2d_cupy(*params)
        if image.ndim == 3:
            result = _warp_3d_cupy(*params)

    result = result.astype(original_dtype, copy=False)

    # from napari import Viewer
    # import napari
    # with napari.gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(image1), name='image_1')
    #     viewer.add_image(_c(image1), name='image_2')

    return result
