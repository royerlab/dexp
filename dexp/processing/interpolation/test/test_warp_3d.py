import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


# def test_warp_2d_numpy():
#     backend = NumpyBackend()
#     _test_warp_2d(backend)


def test_warp_3d_cupy():
    try:
        backend = CupyBackend()
        _test_warp_3d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_warp_3d(backend, length_xy=256, grid_size=8):
    xp = backend.get_xp_module()

    with timeit("generate data"):
        _, _, image = generate_nuclei_background_data(backend,
                                                      add_noise=True,
                                                      length_xy=length_xy,
                                                      length_z_factor=1,
                                                      zoom=2)

    newimage = image[0:512, 0:511, 0:509]
    image = newimage
    image = backend.to_backend(image)

    print(f"shape={image.shape}")

    vector_field = numpy.random.uniform(low=-5, high=+5, size=(grid_size,) * 3 + (3,))

    with timeit("warp"):
        warped = warp(backend, image, vector_field, vector_field_zoom=4)

    with timeit("dewarp"):
        dewarped = warp(backend, warped, -vector_field, vector_field_zoom=4)

    error = xp.mean(xp.absolute(image - dewarped))
    print(f"error = {error}")

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive', rendering='attenuated_mip')
        viewer.add_image(_c(warped), name='warped', colormap='bop purple', blending='additive', rendering='attenuated_mip')
        viewer.add_image(_c(dewarped), name='dewarped', colormap='bop blue', blending='additive', rendering='attenuated_mip')
        viewer.camera.ndisplay = 3

    assert error < 40
