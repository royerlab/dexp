import numpy
from skimage.data import camera

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.reg_warp_nd import register_warp_nd
from dexp.utils.timeit import timeit


def demo_register_warp_2d_numpy():
    backend = NumpyBackend()
    _register_warp_2d(backend)


def demo_register_warp_2d_cupy():
    try:
        backend = CupyBackend()
        _register_warp_2d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _register_warp_2d(backend, warp_grid_size=4, reg_grid_size=8, display=True):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    with timeit("generate dataset"):
        image = camera().astype(numpy.float32)
        image = image[0:510, 0:509]
        image = backend.to_backend(image)

    with timeit("warp"):
        magnitude = 20
        vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 2 + (2,))
        warped = warp(backend, image, vector_field, vector_field_upsampling=4)
        print(f"vector field applied: {vector_field}")

    with timeit("add noise"):
        image += xp.random.uniform(0, 20, size=image.shape)
        warped += xp.random.uniform(0, 20, size=warped.shape)

    with timeit("register_warp_nd"):
        chunks = tuple(s // reg_grid_size for s in image.shape)
        margins = tuple(c // 2 for c in chunks)
        model = register_warp_nd(backend, image, warped, chunks=chunks, margins=margins)
        model.clean(backend)
        print(f"vector field found: {vector_field}")

    with timeit("unwarp"):
        _, unwarped = model.apply(backend, image, warped, vector_field_upsampling=4)

    if display:
        from napari import Viewer, gui_qt
        with gui_qt():
            def _c(array):
                return backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive')
            viewer.add_image(_c(warped), name='warped', colormap='bop blue', blending='additive', visible=False)
            viewer.add_image(_c(unwarped), name='unwarped', colormap='bop purple', blending='additive')

    return image, warped, unwarped, model


if __name__ == "__main__":
    demo_register_warp_2d_cupy()
    # demo_register_warp_2D_numpy()
