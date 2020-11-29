import numpy
from skimage.data import camera

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.interpolation.warp import warp
from dexp.utils.timeit import timeit


# def test_warp_2d_numpy():
#     backend = NumpyBackend()
#     _test_warp_2d(backend)


def test_warp_2d_cupy():
    try:
        backend = CupyBackend()
        _test_warp_2d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_warp_2d(backend, grid_size=8):
    xp = backend.get_xp_module()

    image = camera().astype(numpy.float32) / 255
    image = image[0:377, :]
    image = backend.to_backend(image)

    vector_field = numpy.random.uniform(low=-5, high=+5, size=(grid_size,) * 2 + (2,))

    with timeit("warp"):
        warped = warp(backend, image, vector_field, vector_field_upsampling=4)

    with timeit("dewarp"):
        dewarped = warp(backend, warped, -vector_field, vector_field_upsampling=4)

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

    assert error < 0.03
