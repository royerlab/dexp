import numpy

from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.optimization.grid_search import j_invariant_grid_search
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.deconvolution.admm_deconvolution import admm_deconvolution
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit

from functools import partial


def demo_admm_deconvolution_numpy():
    with NumpyBackend():
        _demo_admm_deconvolution()


def demo_admm_deconvolution_cupy():
    try:
        with CupyBackend():
            _demo_admm_deconvolution()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_admm_deconvolution(length_xy=128):
    xp = Backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(add_noise=False,
                                                                      length_xy=length_xy,
                                                                      length_z_factor=1,
                                                                      background_stength=0,
                                                                      add_offset=False)

    psf = nikon16x08na()
    # psf = olympus20x10na()
    psf = psf.astype(dtype=numpy.float16)
    image = image.astype(dtype=numpy.float16)

    blurry = fft_convolve(image, psf)
    blurry = blurry - blurry.min()
    blurry = blurry / blurry.max()
    noisy = blurry
    noisy = noisy + 0.1 * xp.random.uniform(size=blurry.shape)

    iterations = 20

    # normalizing (required by admm)
    noisy = noisy - noisy.min()
    noisy = noisy / noisy.max()

    deconv = partial(admm_deconvolution, psf=psf, iterations=iterations)
    grid = {
        'rho': numpy.power(10.0, numpy.arange(-2, 4)),
        'gamma': numpy.power(10.0, numpy.arange(-2, 4)),
    }

    def mse(x, y):
        return ((x - y) ** 2).sum().item()

    with timeit("Estimating optimal parameters"):
        params = j_invariant_grid_search(noisy, deconv, mse, grid, display=True)

    with timeit("sparse deconv"):
        deconvolved = deconv(noisy, **params)

    def _c(array):
        return Backend.to_numpy(array)

    import napari
    viewer = napari.Viewer()
    viewer.add_image(_c(image), name='image')
    viewer.add_image(_c(blurry), name='blurry')
    viewer.add_image(_c(psf), name='psf')
    viewer.add_image(_c(noisy), name='noisy')
    viewer.add_image(_c(deconvolved), name='sparse deconv')

    napari.run()



if __name__ == "__main__":
    demo_admm_deconvolution_cupy()
    demo_admm_deconvolution_numpy()
