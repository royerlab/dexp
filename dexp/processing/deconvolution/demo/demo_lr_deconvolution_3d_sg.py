import numpy

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_lr_deconvolution_numpy():
    with NumpyBackend():
        _demo_lr_deconvolution()


def demo_lr_deconvolution_cupy():
    try:
        with CupyBackend():
            _demo_lr_deconvolution()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_lr_deconvolution(length_xy=256):
    with timeit("generate data"):
        _, _, image = generate_nuclei_background_data(
            add_noise=False, length_xy=length_xy, zoom=2, length_z_factor=1, background_strength=0, add_offset=False
        )

    psf = nikon16x08na()
    # psf = olympus20x10na()
    psf = psf.astype(dtype=numpy.float16)
    image = image.astype(dtype=numpy.float16)

    blurry = fft_convolve(image, psf)
    blurry = blurry - blurry.min()
    blurry = blurry / blurry.max()

    max_correction = 2
    iterations = 20

    with timeit("deconvolved"):
        deconvolved = lucy_richardson_deconvolution(
            blurry, psf, num_iterations=iterations, padding=17, max_correction=max_correction
        )

    with timeit("lucy_richardson_deconvolution (scatter-gather)"):

        def f(_image):
            return lucy_richardson_deconvolution(
                _image, psf, num_iterations=iterations, normalise_input=False, padding=17, max_correction=max_correction
            )

        deconvolved_sg = scatter_gather_i2i(blurry, f, tiles=length_xy // 2, margins=17, normalise=True, to_numpy=True)

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name="image")
        viewer.add_image(_c(blurry), name="blurry")
        viewer.add_image(_c(psf), name="psf")
        viewer.add_image(_c(deconvolved), name="deconvolved")
        viewer.add_image(_c(deconvolved_sg), name="deconvolved_sg")


if __name__ == "__main__":
    # demo_lr_deconvolution_cupy()
    demo_lr_deconvolution_numpy()
