import numpy
from arbol import aprint, asection

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.warp_multiscale_nd import register_warp_multiscale_nd
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_register_warp_3d_ms_numpy():
    with NumpyBackend():
        _register_warp_3d_ms()


def demo_register_warp_3d_ms_cupy():
    try:
        with CupyBackend():
            _register_warp_3d_ms()
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


def _register_warp_3d_ms(length_xy=256, warp_grid_size=3, display=True):
    xp = Backend.get_xp_module()

    with asection("generate dataset"):
        _, _, image = generate_nuclei_background_data(
            add_noise=False,
            length_xy=length_xy,
            length_z_factor=1,
            independent_haze=True,
            sphere=True,
            zoom=2,
            dtype=numpy.float32,
        )

        image = image[0:512, 0:511, 0:509]

    with asection("warp"):
        magnitude = 25
        numpy.random.seed(0)
        vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,))
        warped = warp(image, vector_field, vector_field_upsampling=8)
        aprint(f"vector field applied: {vector_field}")

    with asection("add noise"):
        image += xp.random.uniform(-20, 20, size=warped.shape)
        warped += xp.random.uniform(-20, 20, size=warped.shape)

    with asection("register_warp_multiscale_nd"):
        model = register_warp_multiscale_nd(
            image, warped, num_iterations=5, confidence_threshold=0.3, edge_filter=False
        )

    with asection("unwarp"):
        _, unwarped = model.apply_pair(image, warped)

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

    return image, warped, unwarped, model


if __name__ == "__main__":
    demo_register_warp_3d_ms_cupy()
    # demo_register_warp_nD_numpy()
