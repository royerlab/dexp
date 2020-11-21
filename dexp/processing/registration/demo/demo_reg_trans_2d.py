import numpy
from skimage.data import camera

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.reg_trans_nd import register_translation_nd
from dexp.utils.timeit import timeit


def demo_register_translation_2D_numpy():
    backend = NumpyBackend()
    register_translation_2D(backend)


def demo_register_translation_2D_cupy():
    try:
        backend = CupyBackend()
        register_translation_2D(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def register_translation_2D(backend, length_xy=320):
    image1 = camera().astype(numpy.float32) / 255
    magnitude = 35
    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(1,) * 2 + (2,))

    print(f"vector field applied: {vector_field}")

    with timeit("shift"):
        image2 = warp(backend, image1, vector_field)

    with timeit("register_translation_2d"):
        shifts, error = register_translation_nd(backend, image1, image2).get_shift_and_error()
        print(f"shifts: {shifts}, error: {error}")
        shifts = numpy.asarray(shifts)
        vector_field_found = shifts[numpy.newaxis, numpy.newaxis, ...]

    with timeit("shift back"):
        registered = warp(backend, image2, -vector_field_found)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image1), name='image1', colormap='bop orange', blending='additive')
        viewer.add_image(_c(image2), name='image2', colormap='bop blue', blending='additive')
        viewer.add_image(_c(registered), name='registered', colormap='bop purple', blending='additive')


demo_register_translation_2D_cupy()
demo_register_translation_2D_numpy()
