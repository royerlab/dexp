import numpy
from arbol.arbol import aprint

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.deconvolution.admm_deconvolution import admm_deconvolution
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.optimization.grid_search import j_invariant_grid_search
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


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
        image_gt, background, image = generate_nuclei_background_data(
            add_noise=False, length_xy=length_xy, length_z_factor=1, background_strength=0, add_offset=False
        )

    psf = nikon16x08na()
    psf = psf.astype(dtype=numpy.float16)
    image = image.astype(dtype=numpy.float16)

    blurry = fft_convolve(image, psf)
    blurry = blurry - blurry.min()
    blurry = blurry / blurry.max()
    noisy = blurry
    noisy = noisy + 0.1 * xp.random.uniform(size=blurry.shape)

    iterations = 20

    # snippet to test with real data
    # from tifffile import imread
    # psf = nikon16x08na(xy_size=27, z_size=27, dxy=0.485, wvl=0.561, dz=0.94)
    # noisy = imread('/home/jordao/FastData/PhotoM_101421/volume.tiff')

    # normalizing (required by admm)
    noisy = noisy - noisy.min()
    noisy = noisy / noisy.max()

    psf = psf / psf.sum()
    psf = psf.astype("float16")

    def deconv_and_blur(image, rho, gamma):
        deconved = admm_deconvolution(image.astype("float16"), psf, rho, gamma, iterations=iterations, derivative=2)
        return fft_convolve(deconved, psf)

    grid = {
        "rho": numpy.power(10.0, numpy.arange(-3, 3)),
        "gamma": numpy.power(10.0, numpy.arange(-3, 3)),
    }

    def mse(x, y):
        return ((x - y) ** 2).mean().item()

    with timeit("Estimating optimal parameters"):
        params = j_invariant_grid_search(noisy, deconv_and_blur, mse, grid, display=True)

    aprint("Optimal params", params)

    with timeit("sparse deconv"):
        deconvolved = admm_deconvolution(noisy, psf, iterations=iterations, **params, derivative=2)

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
    demo_admm_deconvolution_cupy()
    demo_admm_deconvolution_numpy()
