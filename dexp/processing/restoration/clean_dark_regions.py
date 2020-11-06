import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def clean_dark_regions(backend: Backend,
                       image,
                       size: int = 7,
                       threshold: float = 32,
                       max_proportion_corrected: float = 1,
                       internal_dtype=numpy.float16
                       ):
    """
    Clean Dark Regions

    Filters out noise in dark regions to facilitate compression.


    Parameters
    ----------
    backend : backend to use (numpy, cupy, ...)
    image : image to correct
    num_iterations : number of iterations
    correction_percentile : percentile of pixels to correct per iteration
    """
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = backend.to_backend(image, dtype=internal_dtype, force_copy=True)

    filtered = sp.ndimage.filters.gaussian_filter(image, sigma=2)
    mask = sp.ndimage.filters.maximum_filter(filtered, size=size) < threshold

    num_corrections = xp.sum(mask)

    proportion = num_corrections / image.size

    image[mask] = filtered[mask]

    image = image.astype(original_dtype, copy=False)

    print(
        f"Proportion of denoised pixels: {int(proportion * 100)}% (up to now), versus maximum: {int(max_proportion_corrected * 100)}%) "
    )

    return image
