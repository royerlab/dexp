import numpy
from skimage.data import camera

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp


def demo_warp_numpy():
    backend = NumpyBackend()
    _demo_warp(backend)


def demo_warp_cupy():
    try:
        backend = CupyBackend()
        _demo_warp(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_warp(backend):
    image = camera().astype(numpy.float32) / 255

    vector_field = numpy.asarray([[[10, 10], [-5, 2], [+10, 5]],
                                  [[5, -5], [0, 0], [-10, 2]],
                                  [[5, -1], [1, -5], [-10, -10]]])

    warped = warp(backend, image, vector_field)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(vector_field), name='vector_field')
        viewer.add_image(_c(warped), name='warped')


demo_warp_cupy()
demo_warp_numpy()
