import numpy
from skimage.data import camera
from skimage.util import random_noise

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.butterworth import butterworth_filter


def demo_butterworth_numpy():
    backend = NumpyBackend()
    _demo_butterworth(backend)


def demo_butterworth_cupy():
    try:
        backend = CupyBackend()
        _demo_butterworth(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_butterworth(backend):
    image = camera().astype(numpy.float32) / 255
    noisy = random_noise(image.copy(), mode="gaussian", var=0.005, seed=0, clip=False)
    # noisy = random_noise(noisy, mode="s&p", amount=0.03, seed=0, clip=False)

    filtered = butterworth_filter(backend, noisy, cutoffs=0.3)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(noisy), name='noisy')
        viewer.add_image(_c(filtered), name='filtered')


demo_butterworth_cupy()
demo_butterworth_numpy()
