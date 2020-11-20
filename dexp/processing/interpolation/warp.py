import numpy
from scipy.ndimage import zoom

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def warp(backend: Backend,
         image,
         vector_field,
         vector_field_zoom: float = 2,
         vector_field_zoom_order: int = 2,
         internal_dtype=None):
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

    if not ( image.ndim+1 == vector_field.ndim or (image.ndim == 1 and vector_field.ndim == 1) ):
        raise ValueError("Vector field must have one additional dimension")

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = backend.to_backend(image, dtype=internal_dtype)

    if vector_field_zoom != 1:
        vector_field = backend.to_numpy(vector_field, dtype=internal_dtype)
        vector_field = zoom(vector_field, zoom=(vector_field_zoom,)*image.ndim+(1,), order=vector_field_zoom_order)
    vector_field = backend.to_backend(vector_field, dtype=internal_dtype)

    ## something happens here:
    from dexp.processing.backends.cupy_backend import CupyBackend
    if type(backend) is NumpyBackend:
        raise NotImplementedError("Warping not yet implemented for the Numpy backend.")
    elif type(backend) is CupyBackend:
        params = (backend, image, vector_field)
        if image.ndim == 1:
            from dexp.processing.interpolation._cupy.warp_1d import _warp_1d_cupy
            result = _warp_1d_cupy(*params)
        if image.ndim == 2:
            from dexp.processing.interpolation._cupy.warp_2d import _warp_2d_cupy
            result = _warp_2d_cupy(*params)
        if image.ndim == 3:
            from dexp.processing.interpolation._cupy.warp_3d import _warp_3d_cupy
            result = _warp_3d_cupy(*params)
        else:
            raise NotImplemented("Warping for ndim>3 not implemented.")

    result = result.astype(original_dtype, copy=False)

    return result
