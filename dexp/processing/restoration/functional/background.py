import itertools
from typing import List, Tuple, Any

import numpy
from scipy.ndimage import gaussian_filter

from aydin.corrections.base import ImageCorrectionBase
from aydin.util.log.log import lsection, lprint


class BackgroundCorrection(ImageCorrectionBase):
    """
        Background Correction

        Corrects fixed, axis aligned, offset patterns along any combination of axis.
        Given a list of tuples of axis that defines axis-aligned volumes, intensity fluctuations
        of these volumes are stabilised.
        For example, this this class you can suppress intensity fluctuationa over time,
        suppress fixed offsets per pixel over time, suppress intensity fluctuations per row, per column. etc...
    """

    @staticmethod
    def _axis_combinations(ndim: int, n: int) -> List[Tuple[Any, ...]]:
        return list(itertools.combinations(range(ndim), n))

    def _all_axis_combinations(ndim: int):
        axis_combinations = []
        for dim in range(1, ndim):
            combinations = BackgroundCorrection._axis_combinations(ndim, dim)
            axis_combinations.extend(combinations)
        return axis_combinations

    def __init__(
        self,
        axis_combinations: List[Tuple[int]] = None,
        percentile: float = 1,
        sigma: float = 0.5,
    ):

        """
        Constructs a Background Correction

        Parameters
        ----------
        axis_combinations : List of tuples of axis in the order of correction
        percentile : percentile value used for stabilisation
        sigma : sigma used for Gaussian filtering the computed percentile values.
        """
        super().__init__()

        self.axis_combinations = axis_combinations
        self.percentile = percentile
        self.sigma = sigma

    def correct(self, input_image):
        return self.correct(input_image)

    def uncorrect(self, input_image):
        return self.reapply(input_image)

    def correct(self, array):

        with lsection(f"Background Correction for array of shape: {array.shape}:"):

            original_dtype = array.dtype

            new_array = array.astype(dtype=numpy.float32, copy=True)

            overall_value = numpy.percentile(
                new_array, q=self.percentile, keepdims=True
            )

            axis_combinations = (
                BackgroundCorrection._all_axis_combinations(array.ndim)
                if self.axis_combinations is None
                else self.axis_combinations
            )

            for axis_combination in axis_combinations:
                lprint(f"Supressing variations across hyperplane: {axis_combination}")
                value = numpy.percentile(
                    new_array, q=self.percentile, axis=axis_combination, keepdims=True
                )
                value = gaussian_filter(value, sigma=self.sigma)

                new_array += overall_value - value

            self.overall_value = overall_value

            new_array = new_array.astype(original_dtype, copy=False)

            return new_array

    def reapply(self, array):
        with lsection(f"Background un-suppression for array of shape: {array.shape}:"):
            return array
