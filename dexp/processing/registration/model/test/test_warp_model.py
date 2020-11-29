import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.model.model_factory import from_json
from dexp.processing.registration.model.warp_registration_model import WarpRegistrationModel
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


# TODO: implement numpy version of warp.
# def test_warp_model_numpy():
#     backend = NumpyBackend()
#     _test_warp_model(backend)


def test_warp_model_cupy():
    try:
        backend = CupyBackend()
        _test_warp_model(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _test_warp_model(backend, length_xy=128, warp_grid_size=3):
    xp = backend.get_xp_module()

    _, _, image = generate_nuclei_background_data(backend,
                                                  add_noise=False,
                                                  length_xy=length_xy,
                                                  length_z_factor=1,
                                                  zoom=2,
                                                  dtype=numpy.float32)
    image = image[0:255, 0:253, 0:251]

    magnitude = 15
    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,))
    # print(f"vector field applied: {vector_field}")

    with timeit("warp"):
        image_warped = warp(backend, image, -vector_field, vector_field_upsampling=8)

    model = WarpRegistrationModel(vector_field=vector_field)

    image, image_reg = model.apply(backend, image, image_warped)

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

    json_str = model.to_json()
    new_model = from_json(json_str)
    vf1 = backend.to_numpy(new_model.vector_field)
    vf2 = backend.to_numpy(model.vector_field)
    error = numpy.mean(numpy.absolute(vf1 - vf2))
    print(f"error = {error}")
