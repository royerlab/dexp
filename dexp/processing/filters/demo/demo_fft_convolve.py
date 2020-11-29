import numpy
from scipy.ndimage import convolve
from skimage.data import camera
from skimage.util import random_noise

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.fft_convolve import fft_convolve


def demo_fft_convolve_numpy():
    backend = NumpyBackend()
    _demo_fft_convolve(backend)


def demo_fft_convolve_cupy():
    try:
        backend = CupyBackend()
        _demo_fft_convolve(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_fft_convolve(backend):
    image = camera().astype(numpy.float32) / 255
    noisy = random_noise(image, mode="gaussian", var=0.005, seed=0, clip=False)
    noisy = random_noise(noisy, mode="s&p", amount=0.03, seed=0, clip=False).astype(numpy.float32)

    psf = numpy.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype(numpy.float32)

    result = fft_convolve(backend, noisy, psf)
    reference_result = convolve(noisy, psf)

    result = backend.to_numpy(result)
    reference_result = backend.to_numpy(reference_result)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(noisy), name='noisy')
        viewer.add_image(_c(psf), name='psf')
        viewer.add_image(_c(reference_result), name='reference_result')
        viewer.add_image(_c(result), name='result')


if __name__ == "__main__":
    demo_fft_convolve_cupy()
    demo_fft_convolve_numpy()
