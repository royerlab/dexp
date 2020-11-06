import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.fit_shape import fit_shape


def dehaze(backend: Backend,
           image,
           size: int = 21,
           downscale: int = 2,
           minimal_zero_level: float = 0,
           internal_dtype=numpy.float16
           ):
    sp = backend.get_sp_module()
    xp = backend.get_xp_module()

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = backend.to_backend(image, dtype=internal_dtype)

    minimal_zero_level = backend.to_backend(numpy.asarray(minimal_zero_level), dtype=internal_dtype)

    # get rid of low values due to noise:
    image_zero_level = sp.ndimage.interpolation.zoom(image, zoom=1 / downscale, order=0)

    # find min values:
    image_zero_level = sp.ndimage.filters.minimum_filter(image_zero_level, size=size // 3)

    # expand reach of these min values:
    image_zero_level = sp.ndimage.filters.maximum_filter(image_zero_level, size=size // 3)

    # smooth out:
    image_zero_level = sp.ndimage.filters.gaussian_filter(image_zero_level, sigma=max(1, size // (2 * downscale)))

    # scale up again:
    image_zero_level = sp.ndimage.zoom(image_zero_level, zoom=downscale, order=1)

    # Padding to recover original image size:
    image_zero_level = fit_shape(backend, image_zero_level, shape=image.shape)

    # Ensure that we remove at least the minimum zero level:
    if minimal_zero_level > 0:
        image_zero_level = xp.maximum(image_zero_level, minimal_zero_level)

    # remove zero level:
    image = xp.maximum(image - image_zero_level, 0, out=image)

    # convert back to original dtype
    image = image.astype(dtype=original_dtype, copy=False)

    # from napari import gui_qt, Viewer
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image')
    #     viewer.add_image(_c(image_zero_level), name='image_zero_level')
    #     viewer.add_image(_c(dehazed_image), name='dehazed')

    return image
