import numpy
from skimage.data import camera

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.utils.timeit import timeit


def demo_warp_2d_numpy():
    try:
        backend = NumpyBackend()
        _demo_warp_2d(backend)
    except NotImplementedError:
        print("Numpy version not yet implemented")


def demo_warp_2d_cupy():
    try:
        backend = CupyBackend()
        _demo_warp_2d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_warp_2d(backend, grid_size=8):
    image = camera().astype(numpy.float32) / 255

    vector_field = numpy.random.uniform(low=-15, high=+15, size=(grid_size,)*2+(2,))

    with timeit("warp"):
        warped = warp(backend, image, vector_field, vector_field_zoom=4)

    with timeit("dewarp"):
        dewarped = warp(backend, warped, -vector_field, vector_field_zoom=4)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(vector_field), name='vector_field')
        viewer.add_image(_c(warped), name='warped')
        viewer.add_image(_c(dewarped), name='dewarped')


demo_warp_2d_cupy()
demo_warp_2d_numpy()
