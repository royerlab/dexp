import os

from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF

from dexp.processing.base_restoration import BaseRestoration

os.environ["PYOPENCL"] = "0"


class DecoXXnvolution(BaseRestoration):

    def __init__(self, method='aydin', num_iterations=15, max_correction=8, power=1.5, dxy=0.485, dz=4 * 0.485, xy_size=17, z_size=31):
        """

        """

        psf = SimpleMicroscopePSF()
        psf_xyz_array = psf.generate_xyz_psf(dxy=dxy, dz=dz, xy_size=xy_size, z_size=z_size)
        psf_kernel = psf_xyz_array
        psf_kernel /= psf_kernel.sum()
        self.psf_kernel = psf_kernel
        self.max_num_iterations = num_iterations
        self.max_correction = max_correction
        self.power = power

        self.deconvolver = None
        self.mode = method

    def calibrate(self, images):

        if self.deconvolver is None:
            from aydin.it.deconvolution.llr_deconv import ImageTranslatorLearnedLRDeconv
            self.deconvolver = ImageTranslatorLearnedLRDeconv(backend='cupy',
                                                              psf_kernel=self.psf_kernel,
                                                              max_num_iterations=self.max_num_iterations,
                                                              max_correction=self.max_correction,
                                                              power=self.power,
                                                              normaliser_type='minmax')

        if 'aydin' in self.mode:
            batch_axis = (True,) + (False,) * (images.ndim - 1)
            self.deconvolver.train(images, batch_dims=batch_axis)

    def restore(self, image, asnumpy=True):
        if self.deconvolver is None:
            self.calibrate(image)

        if 'aydin' in self.mode:
            deconvolved_image = self.deconvolver.translate(image)
        elif 'lr' in self.mode:
            deconvolved_image = self.deconvolver.deconvolve(image)

        return deconvolved_image
