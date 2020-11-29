import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.model.warp_registration_model import WarpRegistrationModel
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def demo_warp_model_numpy():
    backend = NumpyBackend()
    warp_model(backend)


def demo_warp_model_cupy():
    try:
        backend = CupyBackend()
        warp_model(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def warp_model(backend, length_xy=320, warp_grid_size=3):
    _, _, image = generate_nuclei_background_data(backend,
                                                  add_noise=False,
                                                  length_xy=length_xy,
                                                  length_z_factor=1,
                                                  zoom=2)

    magnitude = 15
    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,))
    print(f"vector field applied: {vector_field}")

    with timeit("warp"):
        image_warped = warp(backend, image, vector_field, vector_field_upsampling=8)

    model = WarpRegistrationModel(vector_field=vector_field)

    image, image_reg = model.apply(backend, image, image_warped)

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive')
    #     viewer.add_image(_c(image_warped), name='image_warped', colormap='bop purple', blending='additive', visible=False)
    #     viewer.add_image(_c(image_reg), name='image_reg', colormap='bop blue', blending='additive')


if __name__ == "__main__":
    demo_warp_model_cupy()
    demo_warp_model_numpy()
