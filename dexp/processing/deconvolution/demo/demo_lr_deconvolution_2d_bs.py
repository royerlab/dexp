import numpy
from skimage.data import camera
from skimage.util import random_noise

from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.filters.kernels.gaussian import gaussian_kernel_nd
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_lr_deconvolution_numpy():
    with NumpyBackend():
        _demo_lr_deconvolution()


def demo_lr_deconvolution_cupy():
    try:
        with CupyBackend():
            _demo_lr_deconvolution()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_lr_deconvolution():
    image = camera().astype(numpy.float32) / 255

    with NumpyBackend():
        psf = gaussian_kernel_nd(size=9, ndim=2, dtype=numpy.float32)
        blurry = fft_convolve(image, psf)
        blurry = blurry - blurry.min()
        blurry = blurry / blurry.max()
        noisy = random_noise(blurry, mode="gaussian", var=0.01, seed=0, clip=False)
        noisy = random_noise(noisy, mode="s&p", amount=0.01, seed=0, clip=False)

    iterations = 50

    deconvolved = lucy_richardson_deconvolution(noisy, psf, num_iterations=iterations, padding=16)

    deconvolved_blind_spot = lucy_richardson_deconvolution(
        noisy, psf, num_iterations=iterations, padding=16, blind_spot=5, blind_spot_mode="uniform+median"
    )

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name="image")
        viewer.add_image(_c(blurry), name="blurry")
        viewer.add_image(_c(psf), name="psf")
        viewer.add_image(_c(noisy), name="noisy")
        viewer.add_image(_c(deconvolved), name="deconvolved")
        viewer.add_image(_c(deconvolved_blind_spot), name="deconvolved_blind_spot")


if __name__ == "__main__":
    demo_lr_deconvolution_cupy()
    demo_lr_deconvolution_numpy()
