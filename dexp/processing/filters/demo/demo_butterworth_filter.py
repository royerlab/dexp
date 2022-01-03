import numpy
from skimage.data import camera
from skimage.util import random_noise

from dexp.processing.filters.butterworth_filter import butterworth_filter
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_butterworth_filter_numpy():
    with NumpyBackend():
        _demo_butterworth_filter()


def demo_butterworth_filter_cupy():
    try:
        with CupyBackend():
            _demo_butterworth_filter()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_butterworth_filter():
    image = camera().astype(numpy.float32) / 255
    noisy = random_noise(image.copy(), mode="gaussian", var=0.005, seed=0, clip=False)
    # noisy = random_noise(noisy, mode="s&p", amount=0.03, seed=0, clip=False)

    shape = (13, 13)
    cutoffs = 0.75
    n = 8

    filtered = butterworth_filter(noisy, shape=shape, cutoffs=cutoffs, cutoffs_in_freq_units=False, order=n)

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name="image")
        viewer.add_image(_c(noisy), name="noisy")
        viewer.add_image(_c(filtered), name="filtered")


if __name__ == "__main__":
    demo_butterworth_filter_cupy()
    demo_butterworth_filter_numpy()
