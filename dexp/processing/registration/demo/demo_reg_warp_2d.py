import numpy
from skimage.data import camera

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.reg_warp_nd import register_warp_nd
from dexp.utils.timeit import timeit


def demo_register_warp_2D_numpy():
    backend = NumpyBackend()
    register_warp_2D(backend)


def demo_register_warp_2D_cupy():
    try:
        backend = CupyBackend()
        register_warp_2D(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def register_warp_2D(backend, warp_grid_size=1, reg_grid_size=1):
    image = camera().astype(numpy.float32) / 255

    magnitude = 35

    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,)*2+(2,))
    print(f"vector field applied: {vector_field}")

    with timeit("warp"):
        warped = warp(backend, image, vector_field, vector_field_zoom=4)

    with timeit("register_warp_nd"):
        chunks = tuple(s // reg_grid_size for s in image.shape)
        margins = tuple(c//3 for c in chunks)
        model = register_warp_nd(backend, image, warped, chunks=chunks, margins=margins)
        print(f"vector field found: {vector_field}")

    with timeit("unwarp"):
        unwarped = warp(backend, warped, model.vector_field, vector_field_zoom=4)


    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive')
        viewer.add_image(_c(warped), name='warped', colormap='bop blue', blending='additive')
        viewer.add_image(_c(unwarped), name='unwarped', colormap='bop purple', blending='additive')



demo_register_warp_2D_cupy()
#demo_register_warp_2D_numpy()
