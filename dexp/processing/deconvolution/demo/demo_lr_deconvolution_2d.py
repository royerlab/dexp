import numpy
from skimage.data import camera

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

    iterations = 200

    deconvolved_no_noise = lucy_richardson_deconvolution(blurry, psf, num_iterations=iterations, padding=16)

    deconvolved_no_noise_power = lucy_richardson_deconvolution(
        blurry,
        psf,
        num_iterations=iterations,
        padding=16,
        power=1.5,
    )

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name="image")
        viewer.add_image(_c(blurry), name="blurry")
        viewer.add_image(_c(psf), name="psf")
        viewer.add_image(_c(deconvolved_no_noise), name="deconvolved_no_noise")
        viewer.add_image(_c(deconvolved_no_noise_power), name="deconvolved_no_noise_power")


if __name__ == "__main__":
    demo_lr_deconvolution_cupy()
    demo_lr_deconvolution_numpy()
