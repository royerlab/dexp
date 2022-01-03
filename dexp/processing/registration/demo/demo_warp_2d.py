import numpy
from arbol import aprint, asection
from skimage.data import camera

from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.warp_nd import register_warp_nd
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_register_warp_2d_numpy():
    with NumpyBackend():
        _register_warp_2d()


def demo_register_warp_2d_cupy():
    try:
        with CupyBackend():
            _register_warp_2d()
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


def _register_warp_2d(warp_grid_size=4, reg_grid_size=8, display=True):
    xp = Backend.get_xp_module()

    with asection("generate dataset"):
        image = camera().astype(numpy.float32)
        image = image[0:510, 0:509]
        image = Backend.to_backend(image)

    with asection("warp"):
        magnitude = 20
        vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 2 + (2,))
        warped = warp(image, vector_field, vector_field_upsampling=4)
        aprint(f"vector field applied: {vector_field}")

    with asection("add noise"):
        image += xp.random.uniform(0, 20, size=image.shape)
        warped += xp.random.uniform(0, 20, size=warped.shape)

    with asection("register_warp_nd"):
        chunks = tuple(s // reg_grid_size for s in image.shape)
        margins = tuple(c // 2 for c in chunks)
        model = register_warp_nd(image, warped, chunks=chunks, margins=margins)
        model.clean()
        aprint(f"vector field found: {vector_field}")

    with asection("unwarp"):
        _, unwarped = model.apply_pair(image, warped, vector_field_upsampling=4)

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name="image", colormap="bop orange", blending="additive")
            viewer.add_image(_c(warped), name="warped", colormap="bop blue", blending="additive", visible=False)
            viewer.add_image(_c(unwarped), name="unwarped", colormap="bop purple", blending="additive")

    return image, warped, unwarped, model


if __name__ == "__main__":
    demo_register_warp_2d_cupy()
    # demo_register_warp_2D_numpy()
