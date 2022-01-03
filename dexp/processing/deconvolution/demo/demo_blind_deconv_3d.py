from functools import partial

import numpy

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.deconvolution.blind_deconvolution import blind_deconvolution
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_blind_deconvolution_numpy():
    with NumpyBackend():
        _demo_blind_deconvolution()


def demo_blind_deconvolution_cupy():
    try:
        with CupyBackend():
            _demo_blind_deconvolution()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_blind_deconvolution(length_xy=128):
    xp = Backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(
            add_noise=False, length_xy=length_xy, length_z_factor=1, background_strength=0, add_offset=False
        )

    psf = nikon16x08na(xy_size=35, z_size=35)  # , dxy=0.200, dz=0.94)
    psf = psf.astype(dtype=numpy.float16)
    image = image.astype(dtype=numpy.float16)

    blurry = fft_convolve(image, psf)
    blurry = blurry - blurry.min()
    blurry = blurry / blurry.max()
    noisy = blurry
    noisy = noisy + 0.05 * xp.random.uniform(size=blurry.shape)

    # iterations = 10
    # normalizing (required by admm)
    noisy = noisy - noisy.min()
    noisy = noisy / noisy.max()

    # NOTE: could not make it work admm deconvolution (yet)
    # deconv = partial(admm_deconvolution, rho=10, gamma=0.01, iterations=iterations, derivative=2)
    deconv = partial(lucy_richardson_deconvolution, num_iterations=50, padding=psf.shape[2] // 2)
    params = dict(wl=561, na=0.8, ni=1.33, res=485, zres=2000)

    psf = (10000 * psf).astype(int)
    wrong_psf = psf + (psf * 0.1) * numpy.random.uniform(size=psf.shape)
    wrong_psf = psf.astype(int)

    with timeit("Computing blind deconvolution"):
        deconvolved = blind_deconvolution(noisy, wrong_psf, deconv, params, 5)

    def _c(array):
        return Backend.to_numpy(array)

    import napari

    viewer = napari.Viewer()
    viewer.add_image(_c(image), name="image")
    viewer.add_image(_c(blurry), name="blurry")
    viewer.add_image(_c(psf), name="psf")
    viewer.add_image(_c(noisy), name="noisy")
    viewer.add_image(_c(deconvolved), name="sparse deconv")

    napari.run()


if __name__ == "__main__":
    demo_blind_deconvolution_cupy()
    demo_blind_deconvolution_numpy()
