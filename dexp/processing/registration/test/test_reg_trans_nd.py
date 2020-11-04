from pytest import approx

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.reg_trans_nd import register_translation_nd
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.utils.timeit import timeit


def test_register_translation_nD_numpy():
    backend = NumpyBackend()
    register_translation_nD(backend)


def test_register_translation_nD_cupy():
    try:
        backend = CupyBackend()
        register_translation_nD(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_translation_nD(backend, length_xy=128):
    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(backend,
                                                                                       add_noise=False,
                                                                                       shift=(1, 5, -13),
                                                                                       volume_fraction=0.5,
                                                                                       length_xy=length_xy,
                                                                                       length_z_factor=2)

    with timeit("register_translation_nd"):
        shifts, error = register_translation_nd(backend, image1, image2).get_shift_and_error()

    print(shifts, error)

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image, name='array_first')
    #     viewer.add_image(translated_image, name='array_second')

    assert shifts[0] == approx(-1, abs=0.2)
    assert shifts[1] == approx(-5, abs=0.2)
    assert shifts[2] == approx(13, abs=0.2)
