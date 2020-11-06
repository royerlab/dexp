import itertools
from typing import List, Tuple, Any

import numpy

from dexp.processing.backends.backend import Backend


def axis_aligned_pattern_correction(backend: Backend,
                                    image,
                                    axis_combinations: List[Tuple[int]] = None,
                                    percentile: float = 1,
                                    sigma: float = 0.5,
                                    decimation: int = 4,
                                    internal_dtype=numpy.float32,
                                    robust_statistics: bool = True
                                    ):
    """
    Axis aligned pattern correction

    Corrects fixed, axis aligned, offset patterns along any combination of axis.
    Given a list of tuples of axis that defines axis-aligned volumes, intensity fluctuations
    of these volumes are stabilised.
    For example, this this class you can suppress intensity fluctuationa over time,
    suppress fixed offsets per pixel over time, suppress intensity fluctuations per row, per column. etc...


    Parameters
    ----------
    backend : backend to use (numpy, cupy, ...)
    image : image to correct
    axis_combinations : List of tuples of axis in the order of correction
    percentile : percentile value used for stabilisation
    sigma : sigma used for Gaussian filtering the computed percentile values.
    """

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    original_dtype = image.dtype
    new_array = backend.to_backend(image, dtype=internal_dtype, force_copy=True)

    overall_value = numpy.percentile(
        image.ravel()[::decimation], q=percentile, keepdims=True
    )
    overall_value = backend.to_backend(overall_value, dtype=internal_dtype)

    axis_combinations = (
        _all_axis_combinations(image.ndim)
        if axis_combinations is None
        else axis_combinations
    )

    for axis_combination in axis_combinations:
        print(f"Supressing variations across hyperplane: {axis_combination}")
        if robust_statistics:
            value = xp.percentile(
                new_array, q=percentile, axis=axis_combination, keepdims=True
            )
        else:
            value = xp.mean(
                new_array, axis=axis_combination, keepdims=True
            )
        if sigma > 0:
            value = sp.ndimage.filters.gaussian_filter(value, sigma=sigma)

        new_array += overall_value - value

    new_array = new_array.astype(original_dtype, copy=False)

    return new_array


def _axis_combinations(ndim: int, n: int) -> List[Tuple[Any, ...]]:
    return list(itertools.combinations(range(ndim), n))


def _all_axis_combinations(ndim: int):
    axis_combinations = []
    for dim in range(1, ndim):
        combinations = _axis_combinations(ndim, dim)
        axis_combinations.extend(combinations)
    return axis_combinations
