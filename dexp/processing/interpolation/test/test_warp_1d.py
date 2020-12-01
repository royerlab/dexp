import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.interpolation.warp import warp
from dexp.utils.timeit import timeit


# def test_warp_2d_numpy():
#     backend = NumpyBackend()
#     _test_warp_2d(backend)


def test_warp_1d_cupy():
    try:
        with CupyBackend():
            _test_warp_1d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_warp_1d(grid_size=8):
    xp = Backend.get_xp_module()

    image = numpy.random.uniform(low=0, high=1, size=(128,)).astype(dtype=numpy.float32)
    image = Backend.to_backend(image)

    vector_field = numpy.random.uniform(low=-5, high=+5, size=(grid_size,))

    with timeit("warp"):
        warped = warp(image, vector_field, vector_field_upsampling=4)

    with timeit("dewarp"):
        dewarped = warp(warped, -vector_field, vector_field_upsampling=4)

    error = xp.mean(xp.absolute(image - dewarped))
    print(f"error = {error}")

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image')
    #     viewer.add_image(_c(vector_field), name='vector_field')
    #     viewer.add_image(_c(warped), name='warped')
    #     viewer.add_image(_c(dewarped), name='dewarped')

    assert error < 0.3
