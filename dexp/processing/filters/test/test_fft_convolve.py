import numpy
from numpy.linalg import norm
from scipy.ndimage import convolve
from skimage.data import camera
from skimage.util import random_noise

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.fft_convolve import fft_convolve


def test_fft_convolve_numpy():
    with NumpyBackend():
        _test_fft_convolve()


def test_fft_convolve_cupy():
    try:
        with CupyBackend():
            _test_fft_convolve()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_fft_convolve():
    image = camera().astype(numpy.float32) / 255
    noisy = random_noise(image, mode="gaussian", var=0.005, seed=0, clip=False)
    noisy = random_noise(noisy, mode="s&p", amount=0.03, seed=0, clip=False).astype(numpy.float32)

    psf = numpy.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(numpy.float32)

    result = fft_convolve(noisy, psf)
    reference_result = convolve(noisy, psf)

    result = Backend.to_numpy(result)
    reference_result = Backend.to_numpy(reference_result)

    error = norm(reference_result - result, ord=1) / image.size
    print(error)
    assert error < 1e-3
    #
    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image')
    #     viewer.add_image(_c(noisy), name='noisy')
    #     viewer.add_image(_c(psf), name='psf')
    #     viewer.add_image(_c(result), name='result')
