import numpy
from arbol import aprint, asection
from scipy.ndimage import gaussian_filter
from skimage.data import binary_blobs

from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.warp_nd import register_warp_nd
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_register_warp_2d_blobs_numpy():
    with NumpyBackend():
        _register_warp_2d_blobs()


def demo_register_warp_2D_blobs_cupy():
    try:
        with CupyBackend():
            _register_warp_2d_blobs()
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


def _register_warp_2d_blobs(length_xy=512, warp_grid_size=4, reg_grid_size=8, display=True):
    xp = Backend.get_xp_module()

    with asection("generate dataset"):
        image = binary_blobs(length=length_xy, seed=1, n_dim=2, blob_size_fraction=0.04, volume_fraction=0.05)
        image = image.astype(numpy.float32)
        image = gaussian_filter(image, sigma=4)
        image = image[0 : length_xy - 3, 0 : length_xy - 5]
        image = Backend.to_backend(image)

    with asection("warp"):
        magnitude = 20
        vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 2 + (2,))
        warped = warp(image, vector_field, vector_field_upsampling=4)
        aprint(f"vector field applied: {vector_field}")

    with asection("add noise"):
        image += xp.random.uniform(0, 0.1, size=image.shape)
        warped += xp.random.uniform(0, 0.1, size=warped.shape)

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
    demo_register_warp_2D_blobs_cupy()
    # demo_register_warp_2D_numpy()
