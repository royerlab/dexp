import numpy

from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def clean_dark_regions(
    image: xpArray, threshold: float, size: int = 3, mode: str = "uniform", in_place: bool = True, internal_dtype=None
):
    """
    Clean Dark Regions

    Filters out noise in dark regions to facilitate compression.


    Parameters
    ----------
    image : image to correct
    threshold : threshold for 'dark' voxel values.
    size : filter size
    mode : cleaning approach: 'none', 'min', 'uniform', and 'median'
    in_place : True if the input image may be modified in-place.
    internal_dtype : internal dtype for computation

    Returns
    -------
    Cleaned image

    """
    sp = Backend.get_sp_module()

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = Backend.to_backend(image, dtype=internal_dtype, force_copy=not in_place)

    if mode == "none":
        filtered = image.copy()
    elif mode == "min":
        filtered = sp.ndimage.minimum_filter(image, size=size)
    elif mode == "median":
        filtered = sp.ndimage.median_filter(image, size=size)
    elif mode == "uniform":
        filtered = sp.ndimage.uniform_filter(image, size=size)
    else:
        raise ValueError(f"Unknown mode: {mode}, only min, median and uniform supported!")

    mask = sp.ndimage.maximum_filter(filtered, size=size) < threshold

    filtered = sp.ndimage.minimum_filter(image, size=size)

    image[mask] = filtered[mask]

    image = image.astype(original_dtype, copy=False)

    return image
