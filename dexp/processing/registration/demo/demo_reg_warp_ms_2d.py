import numpy
from skimage.data import camera

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.reg_warp_multiscale_nd import register_warp_multiscale_nd
from dexp.utils.timeit import timeit


def demo_register_warp_2d_ms_numpy():
    with NumpyBackend():
        _register_warp_2d_ms()


def demo_register_warp_2d_ms_cupy():
    try:
        with CupyBackend():
            _register_warp_2d_ms()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _register_warp_2d_ms(warp_grid_size=4, reg_grid_size=8, display=True):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    with timeit("generate dataset"):
        image = camera().astype(numpy.float32)
        image = image[0:510, 0:509]
        image = Backend.to_backend(image)

    with timeit("warp"):
        magnitude = 20
        vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 2 + (2,))
        warped = warp(image, vector_field, vector_field_upsampling=4)
        print(f"vector field applied: {vector_field}")

    with timeit("add noise"):
        image += xp.random.uniform(0, 20, size=image.shape)
        warped += xp.random.uniform(0, 20, size=warped.shape)

    with timeit("register_warp_multiscale_nd"):
        model = register_warp_multiscale_nd(image, warped,
                                            num_iterations=5,
                                            confidence_threshold=0.3,
                                            edge_filter=False)

        print(f"vector field found: {model.vector_field}")

    with timeit("unwarp"):
        _, unwarped = model.apply(image, warped, vector_field_upsampling=4)

    if display:
        from napari import Viewer, gui_qt
        with gui_qt():
            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive')
            viewer.add_image(_c(warped), name='warped', colormap='bop blue', blending='additive', visible=False)
            viewer.add_image(_c(unwarped), name='unwarped', colormap='bop purple', blending='additive')

    return image, warped, unwarped, model


if __name__ == "__main__":
    demo_register_warp_2d_ms_cupy()
    # demo_register_warp_2D_numpy()
