import math
from abc import abstractmethod

import numpy
from aydin.it.deconvolution.base_deconv import ImageTranslatorDeconvBase
from aydin.util.log.log import lprint
from aydin.util.offcore.offcore import offcore_array


class ImageTranslatorLRDeconv(ImageTranslatorDeconvBase):
    """
    aydin -- Lucy Richardson Deconvolution Base class

    It's a little bit of a stretch since this is not a 'learned' translation,
    but we can certainly figure out when to stop the iterations based on a provided
    ground truth... The self-supervised case is harder: there are no really good heuristics...

    """

    def __init__(
            self,
            *args,
            max_num_iterations=50,
            max_correction=8,
            power=1,
            back_projection='tpsf',
            no_fft=False,
            **kwargs,
    ):
        """
        Constructs a Lucy Richardson deconvolution image translator.

        Parameters
        ----------
        psf_kernel : array_like
            2D or 3D kernel, dimensions should be odd numbers and numbers sum to 1
        max_num_iterations : int
        clip : bool
        backend : str
            Computation backend selection.
        kwargs : dict
        """
        super().__init__(*args, **kwargs)

        self.power = power
        self.back_projection = back_projection
        self.no_fft = no_fft
        self.max_num_iterations = max_num_iterations
        self.max_correction = max_correction

    def translate(self, input_image, *args, **kwargs):
        self.convolve_method = self._get_convolution_method(
            input_image, self.psf_kernel_numpy
        )
        self.pad_method = self._get_pad_method()

        self.psf_kernel = self._convert_array_format_in(
            self.psf_kernel_numpy.astype(self.dtype)
        )

        if self.back_projection == 'tpsf':
            self.back_projector_numpy = self.psf_kernel[::-1, ::-1]

        elif self.back_projection == 'butterworth':
            cutoffs = (0.2, 1, 1)
            self.back_projector_numpy = self.butterworth(
                psf_kernel=self.psf_kernel_numpy, cutoffs=cutoffs
            )

        self.back_projector = self._convert_array_format_in(self.back_projector_numpy)

        return super().translate(input_image, *args, **kwargs)

    def _translate(self, input_image, image_slice=None, whole_image_shape=None):
        """
        Internal method that translates an input image on the basis of the trained model.

        Parameters
        ----------
        input_image : array_like
            input image
        whole_image_shape : tuple

        Returns
        -------
        deconvolved_image : array_like

        """

        input_image = input_image.astype(self.dtype, copy=False)

        deconvolved_image = offcore_array(shape=input_image.shape, dtype=self.dtype)

        lprint(f"Number of Lucy-Richardson iterations: {self.max_num_iterations}")

        for batch_index, batch_image in enumerate(input_image):

            for channel_index, channel_image in enumerate(batch_image):

                channel_image = channel_image.clip(0, math.inf)
                channel_image = self._convert_array_format_in(channel_image)

                candidate_deconvolved_image = numpy.full(
                    channel_image.shape, float(numpy.mean(channel_image))
                )

                candidate_deconvolved_image = self._convert_array_format_in(
                    candidate_deconvolved_image
                )

                kernel_shape = self.psf_kernel.shape
                pad_width = tuple(
                    (max(self.padding, (s - 1) // 2), max(self.padding, (s - 1) // 2))
                    for s in kernel_shape
                )

                for i in range(self.max_num_iterations):

                    if self.padding > 0:
                        padded_candidate_deconvolved_image = self.pad_method(
                            candidate_deconvolved_image,
                            pad_width=pad_width,
                            mode=self.padding_mode,
                        )
                    else:
                        padded_candidate_deconvolved_image = candidate_deconvolved_image

                    convolved = self.convolve_method(
                        padded_candidate_deconvolved_image,
                        self.psf_kernel,
                        mode='valid' if self.padding else 'same',
                    )

                    relative_blur = channel_image / convolved

                    # zeros = convolved == 0
                    # relative_blur[zeros] = 0

                    relative_blur[
                        relative_blur > self.max_correction
                        ] = self.max_correction
                    relative_blur[relative_blur < 1 / self.max_correction] = (
                            1 / self.max_correction
                    )

                    if self.power != 1.0:
                        relative_blur **= self.power

                    if self.padding:
                        relative_blur = numpy.pad(
                            relative_blur, pad_width=pad_width, mode=self.padding_mode
                        )

                    multiplicative_correction = self.convolve_method(
                        relative_blur,
                        self.back_projector,
                        mode='valid' if self.padding else 'same',
                    )

                    self._debug_allocation(f"after second convolution")

                    candidate_deconvolved_image *= multiplicative_correction

                if self.clip:
                    candidate_deconvolved_image[candidate_deconvolved_image > 1] = 1
                    candidate_deconvolved_image[candidate_deconvolved_image < 0] = 0

                candidate_deconvolved_image = self._convert_array_format_out(
                    candidate_deconvolved_image
                )

                deconvolved_image[
                    batch_index, channel_index
                ] = candidate_deconvolved_image

        return deconvolved_image

    @abstractmethod
    def _convert_array_format_in(self, image):
        raise NotImplemented("This needs ")

    @abstractmethod
    def _convert_array_format_out(self, image):
        raise NotImplemented("This needs ")

    @abstractmethod
    def _get_convolution_method(self, input_image, psf_kernel):
        """
        Method to get convolution method.

        Parameters
        ----------
        input_image : array_like
        psf_kernel : array_like

        Returns
        -------
        convolve

        """
        raise NotImplemented("This needs ")

    def _get_pad_method(self):
        """
        Method to get pad method.

        Returns
        -------
        pad

        """
        raise NotImplemented("This needs ")
