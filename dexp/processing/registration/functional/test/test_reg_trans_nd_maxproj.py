import scipy
from pytest import approx
from skimage.data import binary_blobs
from skimage.filters import gaussian
from skimage.util import random_noise

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.datasets.multiview_data import generate_fusion_test_data
from dexp.processing.registration.functional.reg_trans_2d import register_translation_2d_skimage, register_translation_2d_dexp
from dexp.processing.registration.functional.reg_trans_nd_maxproj import register_translation_maxproj_nd
from dexp.utils.timeit import timeit


def test_register_translation_nD_numpy():
    backend = NumpyBackend()
    register_translation_nD(backend, register_translation_2d_skimage)
    register_translation_nD(backend, register_translation_2d_dexp)

def test_register_translation_nD_cupy():
    try:
        backend = CupyBackend()
        register_translation_nD(backend, register_translation_2d_skimage)
        register_translation_nD(backend, register_translation_2d_dexp)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_translation_nD(backend, reg_trans_2d, length_xy=256):

    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(backend,
                                                                                       add_noise=False,
                                                                                       shift=(1, 5, -13),
                                                                                       volume_fraction=0.5,
                                                                                       length_xy=length_xy,
                                                                                       length_z_factor=2)

    with timeit("register_translation_maxproj_nd"):
        shifts, error = register_translation_maxproj_nd(backend, image1, image2, register_translation_2d=reg_trans_2d)

    print(shifts, error)

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image, name='array_first')
    #     viewer.add_image(translated_image, name='array_second')

    assert shifts[0] == approx(-1, abs=0.5)
    assert shifts[1] == approx(-5, abs=0.5)
    assert shifts[2] == approx(13, abs=0.5)

