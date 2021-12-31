import numpy

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.model.warp_registration_model import (
    WarpRegistrationModel,
)
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_warp_model_numpy():
    with NumpyBackend():
        warp_model()


def demo_warp_model_cupy():
    try:
        with CupyBackend():
            warp_model()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def warp_model(length_xy=320, warp_grid_size=3):
    _, _, image = generate_nuclei_background_data(
        add_noise=False, length_xy=length_xy, length_z_factor=1, zoom=2, dtype=numpy.float32
    )

    magnitude = 15
    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,))
    print(f"vector field applied: {vector_field}")

    with timeit("warp"):
        image_warped = warp(image, vector_field, vector_field_upsampling=8)

    model = WarpRegistrationModel(vector_field=-vector_field)

    image, image_reg = model.apply_pair(image, image_warped)

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name="image", colormap="bop orange", blending="additive")
        viewer.add_image(
            _c(image_warped), name="image_warped", colormap="bop purple", blending="additive", visible=False
        )
        viewer.add_image(_c(image_reg), name="image_reg", colormap="bop blue", blending="additive")


if __name__ == "__main__":
    demo_warp_model_cupy()
    # demo_warp_model_numpy()
