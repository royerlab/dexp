import itertools
from typing import Any, List, Tuple

import numpy

from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def axis_aligned_pattern_correction(
    image: xpArray,
    axis_combinations: List[Tuple[int]] = None,
    quantile: float = 0.5,
    sigma: float = 0,
    decimation: int = 8,
    robust_statistics: bool = True,
    in_place: bool = True,
    internal_dtype=None,
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
    image : image to correct
    axis_combinations : List of tuples of axis in the order of correction
    percentile : percentile value used for stabilisation
    sigma : sigma used for Gaussian filtering the computed percentile values.
    in_place : True if the input image may be modified in-place.
    internal_dtype : internal dtype for computation
    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image.dtype

    image = Backend.to_backend(image, dtype=internal_dtype, force_copy=not in_place)

    axis_combinations = _all_axis_combinations(image.ndim) if axis_combinations is None else axis_combinations

    for axis_combination in axis_combinations:
        print(f"Supressing variations across hyperplane: {axis_combination}")

        overall_value = xp.percentile(image.ravel()[::decimation], q=100 * quantile, keepdims=True)

        if robust_statistics:
            value = xp.percentile(image, q=100 * quantile, axis=axis_combination, keepdims=True)
            value = value.astype(dtype=internal_dtype, copy=False)
        else:
            value = xp.mean(image, axis=axis_combination, keepdims=True, dtype=internal_dtype)

        if sigma > 0:
            value -= sp.ndimage.gaussian_filter(value, sigma=sigma)

        image += overall_value
        image -= value

    image = image.astype(original_dtype, copy=False)

    return image


def _axis_combinations(ndim: int, n: int) -> List[Tuple[Any, ...]]:
    return list(itertools.combinations(range(ndim), n))


def _all_axis_combinations(ndim: int):
    axis_combinations = []
    for dim in range(1, ndim):
        combinations = _axis_combinations(ndim, dim)
        axis_combinations.extend(combinations)
    return axis_combinations
