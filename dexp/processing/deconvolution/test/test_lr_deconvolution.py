import numpy
from skimage.data import camera
from skimage.util import random_noise

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.filters.kernels.gaussian import gaussian_kernel_nd


def test_lr_deconvolution_numpy():
    with NumpyBackend():
        _test_lr_deconvolution()


def test_lr_deconvolution_cupy():
    try:
        with CupyBackend():
            _test_lr_deconvolution()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_lr_deconvolution():
    xp = Backend.get_xp_module()

    image = camera().astype(numpy.float32) / 255
    noisy = random_noise(image, mode="gaussian", var=0.005, seed=0, clip=False)
    noisy = random_noise(noisy, mode="s&p", amount=0.03, seed=0, clip=False)

    psf = gaussian_kernel_nd(size=9, ndim=2, sigma=2, dtype=numpy.float32)

    image = Backend.to_backend(image)
    noisy = Backend.to_backend(noisy)
    psf = Backend.to_backend(psf)

    blurry = fft_convolve(image, psf)

    deconvolved = lucy_richardson_deconvolution(blurry, psf, padding=16)

    error = xp.linalg.norm(deconvolved - image, ord=1) / image.size

    print(f"Error = {error}")

    assert error < 0.001
