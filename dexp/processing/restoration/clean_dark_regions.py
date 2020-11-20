import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def clean_dark_regions(backend: Backend,
                       image,
                       size: int = 7,
                       threshold: float = 32,
                       mode: str = 'gaussian',
                       sigma: float = 1,
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
    size : filter size
    threshold : threshold for 'dark' voxel values.
    mode : cleaning approach: 'min', 'gaussian', and 'median'
    sigma : sigma value for gaussian filtering case.
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

    if mode == 'min':
        filtered = sp.ndimage.filters.minimum_filter(image, size=3)
    elif mode == 'median':
        filtered = sp.ndimage.filters.median_filter(image, size=3)
    elif mode == 'gaussian':
        filtered = sp.ndimage.filters.gaussian_filter(image, sigma=sigma)
    else:
        raise ValueError('Unknown mode')

    mask = sp.ndimage.filters.maximum_filter(filtered, size=size) < threshold
    # num_corrections = xp.sum(mask)
    # proportion = num_corrections / image.size

    image[mask] = filtered[mask]

    image = image.astype(original_dtype, copy=False)

    # print(
    #     f"Proportion of denoised pixels: {int(proportion * 100)}% (up to now), versus maximum: {int(max_proportion_corrected * 100)}%) "
    # )

    return image
