import numpy
from scipy.ndimage import median_filter

from aydin.corrections.base import ImageCorrectionBase
from aydin.util.log.log import lsection, lprint


class BrokenPixelsCorrection(ImageCorrectionBase):
    """
        Broken Pixels Correction

        'Broken' pixels on detectors typically blink or are very dim or very bright, in any case they are 'out of context'.
        While self-supervised denoising can solve many of these issues, in some cases these pixels values are predictable,
        and thus are not denoised.
    """

    def __init__(
        self,
        num_iterations: int = 2,
        correction_percentile: float = 0.1,
        lipschitz: float = 0.1,
        max_proportion_corrected: float = 1,
    ):

        """
        Constructs a Broken Pixels Corrector

        Parameters
        ----------
        num_iterations : number of iterations
        correction_percentile : percentile of pixels to correct per iteration
        lipschitz : lipschitz continuity constant
        max_proportion_corrected : max proportion of pixels to correct overall
        """
        super().__init__()

        self.num_iterations = num_iterations
        self.correction_percentile = correction_percentile
        self.lipschitz = lipschitz
        self.max_proportion_corrected = max_proportion_corrected

    def correct(self, input_image):
        return self.correct(input_image)

    def uncorrect(self, input_image):
        return input_image

    def correct(self, array):

        with lsection(f"Broken Pixels Correction for array of shape: {array.shape}:"):

            original_dtype = array.dtype
            array = array.astype(dtype=numpy.float32, copy=True)

            total_number_of_corrections = 0

            for i in range(self.num_iterations):
                lprint(f"Iteration {i}")
                median, error = self.compute_error(array)
                threshold = numpy.percentile(
                    error, q=100 * (1 - self.correction_percentile)
                )

                mask = error > threshold

                num_corrections = numpy.sum(mask)
                lprint(f"Number of corrections: {num_corrections}")

                if num_corrections == 0:
                    break

                proportion = (
                    num_corrections + total_number_of_corrections
                ) / array.size
                lprint(
                    f"Proportion of corrected pixels: {int(proportion * 100)}% (up to now), versus maximum: {int(self.max_proportion_corrected * 100)}%) "
                )

                if proportion > self.max_proportion_corrected:
                    break

                array[mask] = median[mask]

                total_number_of_corrections += num_corrections

            array = array.astype(original_dtype, copy=False)

            return array

    def compute_error(self, array):
        # we compute the error map:
        median = median_filter(array, size=3)
        error = median.copy()
        error -= array
        numpy.abs(error, out=error)
        numpy.maximum(error, self.lipschitz, out=error)
        error -= self.lipschitz
        return median, error
