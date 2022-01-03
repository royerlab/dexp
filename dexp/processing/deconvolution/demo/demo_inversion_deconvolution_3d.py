import numpy

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.deconvolution.inversion_deconvolution import (
    inversion_deconvolution,
)
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_inversion_deconvolution_numpy():
    with NumpyBackend():
        _demo_inversion_deconvolution()


def demo_inversion_deconvolution_cupy():
    try:
        with CupyBackend():
            _demo_inversion_deconvolution()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_inversion_deconvolution(length_xy=128):
    xp = Backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(
            add_noise=False, length_xy=length_xy, length_z_factor=1, background_strength=0, add_offset=False
        )

    psf = nikon16x08na()
    # psf = olympus20x10na()
    psf = psf.astype(dtype=numpy.float16)
    image = image.astype(dtype=numpy.float16)

    blurry = fft_convolve(image, psf)
    blurry = blurry - blurry.min()
    blurry = blurry / blurry.max()
    noisy = blurry
    noisy = noisy + 0.1 * xp.random.uniform(size=blurry.shape)

    with timeit("deconvolved"):
        deconvolved = inversion_deconvolution(
            noisy, psf, mode="reflect", epsilon=1e-1
        )  # FIXME noisy, psf, mode='reflect')

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


if __name__ == "__main__":
    demo_inversion_deconvolution_cupy()
    demo_inversion_deconvolution_numpy()
