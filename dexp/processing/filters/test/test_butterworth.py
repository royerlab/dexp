import numpy
from skimage.data import camera
from skimage.util import random_noise

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.butterworth_filter import butterworth_filter


def test_butterworth_numpy():
    with NumpyBackend():
        _test_butterworth()


def test_butterworth_cupy():
    try:
        with CupyBackend():
            _test_butterworth()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_butterworth():
    image = camera().astype(numpy.float32) / 255
    noisy = random_noise(image.copy(), mode="gaussian", var=0.005, seed=0, clip=False)
    # noisy = random_noise(noisy, mode="s&p", amount=0.03, seed=0, clip=False)

    filtered = butterworth_filter(noisy)

    assert filtered.shape == noisy.shape
    assert filtered.dtype == noisy.dtype

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(noisy, name='noisy')
    #     viewer.add_image(filtered, name='filtered')
