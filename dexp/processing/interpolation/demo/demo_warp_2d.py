import numpy
from skimage.data import camera

from dexp.processing.interpolation.warp import warp
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_warp_2d_numpy():
    try:
        with NumpyBackend():
            _demo_warp_2d()
    except NotImplementedError:
        print("Numpy version not yet implemented")


def demo_warp_2d_cupy():
    try:
        with CupyBackend():
            _demo_warp_2d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_warp_2d(grid_size=8):
    image = camera().astype(numpy.float32) / 255
    image = image[0:477, 0:507]

    magnitude = 15
    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(grid_size,) * 2 + (2,))

    with timeit("warp"):
        warped = warp(image, vector_field, vector_field_upsampling=4)

    with timeit("dewarped"):
        dewarped = warp(warped, -vector_field, vector_field_upsampling=4)

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name="image")
        viewer.add_image(_c(vector_field), name="vector_field")
        viewer.add_image(_c(warped), name="warped")
        viewer.add_image(_c(dewarped), name="dewarped")


if __name__ == "__main__":
    demo_warp_2d_cupy()
    # demo_warp_2d_numpy()
