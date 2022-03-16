import numpy

from dexp.processing.utils.fit_shape import fit_to_shape
from dexp.processing.utils.nan_to_zero import nan_to_zero
from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def dehaze(
    image: xpArray,
    size: int = 21,
    downscale: int = 4,
    minimal_zero_level: float = 0,
    correct_max_level: bool = True,
    in_place: bool = True,
    internal_dtype=None,
):
    """
    Dehazes an image by means of a non-linear low-pass rejection filter.

    Parameters
    ----------
    image : image to filter
    size : filter size
    downscale : downscale factor for speeding up computation of the haze map.
    minimal_zero_level : minimal zero level to substract
    in_place : True if the input image may be modified in-place.
    internal_dtype : internal dtype for computation

    Returns
    -------
    Dehazed image

    """
    sp = Backend.get_sp_module()
    xp = Backend.get_xp_module()

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = Backend.to_backend(image, dtype=internal_dtype, force_copy=not in_place)
    # original_image = image.copy()

    minimal_zero_level = xp.asarray(minimal_zero_level, dtype=internal_dtype)

    # get rid of low values due to noise:
    image_zero_level = sp.ndimage.maximum_filter(image, size=3)

    # downscale to speed up the rest of the computation:
    downscaled_image = sp.ndimage.zoom(image_zero_level, zoom=1 / downscale, order=0)

    # find min values:
    image_zero_level = sp.ndimage.minimum_filter(downscaled_image, size=max(1, size // downscale))

    # expand reach of these min values:
    image_zero_level = sp.ndimage.maximum_filter(image_zero_level, size=max(1, size // downscale))

    # smooth out:
    image_zero_level = sp.ndimage.gaussian_filter(image_zero_level, sigma=max(1, size // (2 * downscale)))

    # scale up again:
    image_zero_level = sp.ndimage.zoom(image_zero_level, zoom=downscale, order=1)

    # Padding to recover original image size:
    image_zero_level = fit_to_shape(image_zero_level, shape=image.shape)

    # Ensure that we remove at least the minimum zero level:
    if minimal_zero_level > 0:
        image_zero_level = xp.maximum(image_zero_level, minimal_zero_level)

    # remove zero level:
    image -= image_zero_level
    del image_zero_level

    # clip:
    image = xp.maximum(image, 0, out=image)

    if correct_max_level:
        # get image max level before:
        # twice filtering is to match the extent reached for the zero_level image
        # (see above combination of min then max filters)
        downscaled_image = sp.ndimage.maximum_filter(downscaled_image, size=max(1, size // downscale))
        image_max_level_before = sp.ndimage.gaussian_filter(downscaled_image, sigma=max(1, size // downscale))

        # get image max level after:
        downscaled_image_after = sp.ndimage.maximum_filter(image, size=3)
        downscaled_image_after = sp.ndimage.zoom(downscaled_image_after, zoom=1 / downscale, order=0)
        image_max_level_after = sp.ndimage.maximum_filter(downscaled_image_after, size=max(1, size // downscale))
        image_max_level_after = sp.ndimage.gaussian_filter(image_max_level_after, sigma=max(1, size // downscale))

        # Correction ratio:
        epsilon = xp.asarray(1e-6, dtype=internal_dtype)
        correction_ratio = image_max_level_before
        correction_ratio /= image_max_level_after + epsilon
        correction_ratio = nan_to_zero(correction_ratio, copy=False)
        del image_max_level_after
        correction_ratio = sp.ndimage.zoom(correction_ratio, zoom=downscale, order=1)
        correction_ratio = fit_to_shape(correction_ratio, shape=image.shape)

        image *= correction_ratio
        del correction_ratio

    # convert back to original dtype
    image = image.astype(dtype=original_dtype, copy=False)

    # from napari import gui_qt, Viewer
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='original_image')
    #     viewer.add_image(_c(image_zero_level), name='image_zero_level')
    #     viewer.add_image(_c(image), name='dehazed')

    return image
