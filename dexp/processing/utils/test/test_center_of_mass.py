import numpy
from arbol import asection, aprint

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.processing.utils.center_of_mass import center_of_mass


def test_com_numpy():
    with NumpyBackend():
        _test_com()


def test_com_cupy():
    try:
        with CupyBackend():
            _test_com()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_com(length_xy=128):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    with asection("generate data"):
        _, _, image = generate_nuclei_background_data(add_noise=False,
                                                      length_xy=length_xy,
                                                      length_z_factor=1,
                                                      background_stength=0.001,
                                                      sphere=True,
                                                      radius=0.5,
                                                      zoom=2,
                                                      add_offset=False,
                                                      dtype=numpy.uint16)

        # from napari import Viewer, gui_qt
        # with gui_qt():
        #     def _c(array):
        #         return Backend.to_numpy(array)
        #     viewer = Viewer()
        #     viewer.add_image(_c(image), name='image')

        com_before = center_of_mass(image)
        aprint(f"com_before: {com_before}")

        image_shifted = sp.ndimage.shift(image, shift=(50, 70, -23), order=1, mode='nearest')

        com_after = center_of_mass(image_shifted)
        aprint(f"com_after: {com_after}")

        com_after_bb = center_of_mass(image_shifted, offset_mode='p=75', bounding_box=True)
        aprint(f"com_after_bb: {com_after_bb}")

        # from napari import Viewer, gui_qt
        # with gui_qt():
        #     def _c(array):
        #         return Backend.to_numpy(array)
        #     viewer = Viewer()
        #     viewer.add_image(_c(image), name='image')
        #     viewer.add_image(_c(image_shifted), name='image_shifted')

        assert xp.mean(xp.absolute((com_after - xp.asarray((50, 70, -23)) - com_before))) < 10
        assert xp.mean(xp.absolute((com_after_bb - xp.asarray((50, 70, -23)) - com_before))) < 10
