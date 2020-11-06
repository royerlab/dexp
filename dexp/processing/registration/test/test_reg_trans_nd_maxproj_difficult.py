from pytest import approx

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.reg_trans_2d import register_translation_2d_dexp
from dexp.processing.registration.reg_trans_nd_maxproj import register_translation_maxproj_nd
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.utils.timeit import timeit


def test_register_translation_nD_numpy():
    backend = NumpyBackend()
    register_translation_nD(backend, register_translation_2d_dexp)

    # Lesson: skimage registration code is as robust!
    # register_translation_nD(backend, register_translation_2d_skimage)


def test_register_translation_nD_cupy():
    try:
        backend = CupyBackend()
        register_translation_nD(backend, register_translation_2d_dexp)

        # Lesson: skimage registration code is as robust!
        # register_translation_nD(backend, register_translation_2d_skimage)

    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_translation_nD(backend, reg_trans_2d, length_xy=128):
    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(backend,
                                                                                       add_noise=False,
                                                                                       shift=(1, 5, -13),
                                                                                       volume_fraction=0.5,
                                                                                       length_xy=length_xy,
                                                                                       length_z_factor=2,
                                                                                       z_overlap=1)
    depth = image1.shape[0]
    crop = depth // 4

    image1_c = image1[crop:-crop]
    image2_c = image2[crop:-crop]

    with timeit("register_translation_maxproj_nd"):
        shifts, error = register_translation_maxproj_nd(backend, image1_c, image2_c, register_translation_2d=reg_trans_2d).get_shift_and_error()

    print(shifts, error)

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image_gt), name='image_gt')
    #     viewer.add_image(_c(image_lowq), name='image_lowq')
    #     viewer.add_image(_c(blend_a), name='blend_a')
    #     viewer.add_image(_c(blend_b), name='blend_b')
    #     viewer.add_image(_c(image1), name='image1')
    #     viewer.add_image(_c(image2), name='image2')

    assert shifts[0] == approx(-1, abs=0.5)
    assert shifts[1] == approx(-5, abs=0.5)
    assert shifts[2] == approx(13, abs=0.5)
