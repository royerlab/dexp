import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.model.warp_registration_model import WarpRegistrationModel
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


# TODO: implement numpy version of warp.
# def test_warp_model_numpy():
#     backend = NumpyBackend()
#     _test_warp_model(backend)


def test_warp_model_cupy():
    try:
        with CupyBackend():
            _test_warp_model()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _test_warp_model(length_xy=128, warp_grid_size=3):
    xp = Backend.get_xp_module()

    _, _, image = generate_nuclei_background_data(add_noise=False,
                                                  length_xy=length_xy,
                                                  length_z_factor=1,
                                                  zoom=2,
                                                  dtype=numpy.float32)
    image = image[0:255, 0:253, 0:251]

    magnitude = 15
    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,))
    confidence = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3)
    # print(f"vector field applied: {vector_field}")

    with timeit("warp"):
        image_warped = warp(image, -vector_field, vector_field_upsampling=8)

    model = WarpRegistrationModel(vector_field=vector_field, confidence=confidence)

    image, image_reg = model.apply_pair(image, image_warped)

    warped_error = xp.mean(xp.absolute(image - image_warped))
    dewarped_error = xp.mean(xp.absolute(image - image_reg))
    print(f"warped_error = {warped_error}")
    print(f"dewarped_error = {dewarped_error}")

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive', rendering='attenuated mip')
    #     viewer.add_image(_c(image_warped), name='image_warped', colormap='bop purple', blending='additive', visible=False, rendering='attenuated mip')
    #     viewer.add_image(_c(image_reg), name='image_reg', colormap='bop blue', blending='additive', rendering='attenuated mip')
    #     #viewer.camera.ndisplay=3

    assert dewarped_error < warped_error
    assert dewarped_error < 55
