import numpy
import scipy
from arbol import aprint, asection

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.warp_nd import register_warp_nd
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_register_warp_3d_numpy():
    with NumpyBackend():
        _register_warp_3d()


def demo_register_warp_3d_cupy():
    try:
        with CupyBackend():
            _register_warp_3d()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _register_warp_3d(length_xy=256, warp_grid_size=3, reg_grid_size=6, display=True):
    xp = Backend.get_xp_module()

    _, _, image = generate_nuclei_background_data(
        add_noise=False,
        length_xy=length_xy,
        length_z_factor=1,
        independent_haze=True,
        sphere=True,
        zoom=2,
        dtype=numpy.float32,
    )

    image = image[0 : length_xy * 2 - 3, 0 : length_xy * 2 - 5, 0 : length_xy * 2 - 7]

    with asection("warp"):
        magnitude = 10
        vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,))
        warped = warp(image, vector_field, vector_field_upsampling=8)
        aprint(f"vector field applied: {vector_field}")

    with asection("add noise"):
        image += xp.random.uniform(0, 40, size=image.shape)
        warped += xp.random.uniform(0, 40, size=warped.shape)

    with asection("register_warp_nd"):
        chunks = tuple(s // reg_grid_size for s in image.shape)
        margins = tuple(max(4, c // 3) for c in chunks)
        aprint(f"chunks={chunks}, margins={margins}")
        model = register_warp_nd(image, warped, chunks=chunks, margins=margins)
        model.clean()
        # print(f"vector field found: {vector_field}")

    with asection("unwarp"):
        _, unwarped = model.apply_pair(image, warped, vector_field_upsampling=4)

    vector_field = scipy.ndimage.zoom(vector_field, zoom=(2, 2, 2, 1), order=1)

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(
                _c(image),
                name="image",
                colormap="bop orange",
                blending="additive",
                rendering="attenuated_mip",
                attenuation=0.01,
            )
            viewer.add_image(
                _c(warped),
                name="warped",
                colormap="bop blue",
                blending="additive",
                visible=False,
                rendering="attenuated_mip",
                attenuation=0.01,
            )
            viewer.add_image(
                _c(unwarped),
                name="unwarped",
                colormap="bop purple",
                blending="additive",
                rendering="attenuated_mip",
                attenuation=0.01,
            )
            viewer.add_vectors(_c(vector_field), name="gt vector_field")
            viewer.add_vectors(_c(model.vector_field), name="model vector_field")

    return image, warped, unwarped, model


if __name__ == "__main__":
    demo_register_warp_3d_cupy()
    # demo_register_warp_3d_numpy()
