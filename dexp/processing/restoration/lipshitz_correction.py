import gc

import numpy

from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def lipschitz_continuity_correction(
    image: xpArray,
    num_iterations: int = 2,
    correction_percentile: float = 0.1,
    lipschitz: float = 0.1,
    max_proportion_corrected: float = 1,
    decimation: int = 8,
    in_place: bool = True,
    internal_dtype=None,
):
    """
    'Lipshitz continuity correction'

    'Broken' pixels on detectors typically blink or are very dim or very bright, in any case they are 'out of context'.
    In many cases they will locally break Lipshitz continuity implied by diffraction limited imaging.
    Here is a simple greedy scheme that starts with the ost infringing voxels
    and incrementally filters them using local median filtering.


    Parameters
    ----------

    image : image to correct
    num_iterations : number of iterations
    correction_percentile : percentile of pixels to correct per iteration
    lipschitz : lipschitz continuity constant
    max_proportion_corrected : max proportion of pixels to correct overall
    decimation : decimation for speeding up percentile computation
    in_place : True if the input image may be modified in-place.
    internal_dtype : internal dtype

    Returns
    -------
    Lipschitz corrected image

    """

    xp = Backend.get_xp_module()

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype
    image = Backend.to_backend(image, dtype=internal_dtype, force_copy=not in_place)

    total_number_of_corrections = 0

    for i in range(num_iterations):
        print(f"Iteration {i}")
        # TODO: it is slow to recompute the median filter at each iteration,
        # could be done only once but that's less accurate..
        median, error = _compute_error(image, decimation, lipschitz, internal_dtype)
        gc.collect()
        threshold = xp.percentile(error.ravel()[::decimation], q=100 * (1 - correction_percentile))

        mask = error > threshold

        num_corrections = xp.sum(mask)
        print(f"Number of corrections: {num_corrections}")

        if num_corrections == 0:
            break

        proportion = (num_corrections + total_number_of_corrections) / image.size
        print(
            f"Proportion of corrected pixels: {int(proportion * 100)}% (up to now), "
            + f"versus maximum: {int(max_proportion_corrected * 100)}%)"
        )

        if proportion > max_proportion_corrected:
            break

        image[mask] = median[mask]

        total_number_of_corrections += num_corrections

        gc.collect()

    array = image.astype(original_dtype, copy=False)

    return array


def _compute_error(array, decimation: int, lipschitz: float, dtype=numpy.float16):
    sp = Backend.get_sp_module()
    xp = Backend.get_xp_module()
    array = Backend.to_backend(array, dtype=dtype)
    # we compute the error map:
    median = sp.ndimage.median_filter(array, size=3)
    error = median.copy()
    error -= array
    xp.abs(error, out=error)
    xp.maximum(error, lipschitz, out=error)
    error -= lipschitz
    return median, error
