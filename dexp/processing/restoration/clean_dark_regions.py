import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def clean_dark_regions(backend: Backend,
                       image,
                       threshold: float,
                       size: int = 5,
                       mode: str = 'median',
                       in_place: bool = True,
                       internal_dtype=None
                       ):
    """
    Clean Dark Regions

    Filters out noise in dark regions to facilitate compression.


    Parameters
    ----------
    backend : backend to use (numpy, cupy, ...)
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

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = backend.to_backend(image, dtype=internal_dtype, force_copy=not in_place)

    if mode == 'none':
        filtered = image.copy()
    elif mode == 'min':
        filtered = sp.ndimage.filters.minimum_filter(image, size=size)
    elif mode == 'median':
        filtered = sp.ndimage.filters.median_filter(image, size=size)
    elif mode == 'uniform':
        filtered = sp.ndimage.filters.uniform_filter(image, size=size)
    else:
        raise ValueError(f'Unknown mode: {mode}, only min, median and uniform supported!')

    mask = sp.ndimage.filters.maximum_filter(filtered, size=size) < threshold

    filtered = sp.ndimage.filters.minimum_filter(image, size=size)

    image[mask] = filtered[mask]

    image = image.astype(original_dtype, copy=False)

    return image
