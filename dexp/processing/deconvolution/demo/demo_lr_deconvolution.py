import numpy
from numpy.linalg import norm
from scipy.ndimage import convolve
from skimage.data import camera
from skimage.util import random_noise

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.fft_convolve import fft_convolve

def gaussian_kernel(length=5, sigma=1.):
    ax = numpy.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    xx, yy = numpy.meshgrid(ax, ax)
    kernel = numpy.exp(-0.5 * (numpy.square(xx) + numpy.square(yy)) / numpy.square(sigma))
    return kernel / numpy.sum(kernel)

def demo_lr_deconvolution_numpy():
    backend = NumpyBackend()
    _demo_lr_deconvolution(backend)


def demo_lr_deconvolution_cupy():
    try:
        backend = CupyBackend()
        _demo_lr_deconvolution(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_lr_deconvolution(backend):
    image = camera().astype(numpy.float32) / 255
    noisy = random_noise(image, mode="gaussian", var=0.005, seed=0, clip=False)
    noisy = random_noise(noisy, mode="s&p", amount=0.03, seed=0, clip=False)

    psf = gaussian_kernel(9, 2).astype(numpy.float32)

    result = fft_convolve(backend, image, psf)




    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(noisy), name='noisy')
        viewer.add_image(_c(psf), name='psf')
        viewer.add_image(_c(result), name='result')


demo_lr_deconvolution_cupy()
demo_lr_deconvolution_numpy()
