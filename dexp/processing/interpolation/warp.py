import numpy
from scipy.ndimage import zoom

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def warp(backend: Backend,
         image,
         vector_field,
         vector_field_upsampling: int = 2,
         vector_field_upsampling_order: int = 1,
         mode: str = 'border',
         internal_dtype=None):
    """
    Applies a warp transform (piece wise linear or constant) to an image based on a vector field.
    Only implemented for 1d, 2d, and 3d images.

    Parameters
    ----------
    backend : backend to use
    image : image to warp
    vector_field : vector field to warp inoput image with. The vector field is an array of dimension n+1 where n is the dimension of the input image.
    The first n dimensions can be of arbirary lengths, and the last vector is the warp vector for each image region that the first
    vector_field_upsampling : upsampling factor for teh vector field (best use a power of two)
    vector_field_upsampling_order : upsampling order: 0-> nearest, 1->linear, 2->quadratic, ... (uses scipy zoom)
    mode : How to handle warping that reaches outside of the image bounds, can be: 'clamp', 'border', 'wrap', 'mirror'
    internal_dtype : internal dtype

    Returns
    -------
    Warped image

    """

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    if not (image.ndim + 1 == vector_field.ndim or (image.ndim == 1 and vector_field.ndim == 1)):
        raise ValueError("Vector field must have one additional dimension")

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype

    if vector_field_upsampling != 1:
        # Note: unfortunately numpy does support float16 zooming, and cupy does not support high-order zooming...
        vector_field = backend.to_numpy(vector_field, dtype=numpy.float32)
        if image.ndim > 1:
            vector_field = zoom(vector_field, zoom=(vector_field_upsampling,) * image.ndim + (1,), order=vector_field_upsampling_order)
        else:
            vector_field = zoom(vector_field, zoom=(vector_field_upsampling,), order=vector_field_upsampling_order)

    image = backend.to_backend(image, dtype=internal_dtype)
    vector_field = backend.to_backend(vector_field, dtype=internal_dtype)

    from dexp.processing.backends.cupy_backend import CupyBackend
    if type(backend) is NumpyBackend:

        raise NotImplementedError("Warping not yet implemented for the Numpy backend.")
    elif type(backend) is CupyBackend:

        params = (backend, image, vector_field, mode)
        if image.ndim == 1:
            from dexp.processing.interpolation._cupy.warp_1d import _warp_1d_cupy
            result = _warp_1d_cupy(*params)
        elif image.ndim == 2:
            from dexp.processing.interpolation._cupy.warp_2d import _warp_2d_cupy
            result = _warp_2d_cupy(*params)
        elif image.ndim == 3:
            from dexp.processing.interpolation._cupy.warp_3d import _warp_3d_cupy
            result = _warp_3d_cupy(*params)
        else:
            raise NotImplementedError("Warping for ndim>3 not implemented.")

    result = result.astype(original_dtype, copy=False)

    return result
