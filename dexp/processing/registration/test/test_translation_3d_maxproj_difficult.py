from pytest import approx

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.demo.demo_translation_3d_proj_difficult import _register_translation_3d_proj_diff
from dexp.processing.registration.translation_2d import register_translation_2d_dexp


def test_register_translation_3d_maxproj_diff_numpy():
    with NumpyBackend():
        register_translation_3d_maxproj_diff(register_translation_2d_dexp)

    # Lesson: skimage registration code is not as robust!
    # register_translation_nD(backend, register_translation_2d_skimage)


def test_register_translation_3d_maxproj_diff_cupy():
    try:
        with CupyBackend():
            register_translation_3d_maxproj_diff(register_translation_2d_dexp)

        # Lesson: skimage registration code is not as robust!
        # register_translation_nD(backend, register_translation_2d_skimage)

    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_translation_3d_maxproj_diff(reg_trans_2d, length_xy=128):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    image, shifted, unshifted, model = _register_translation_3d_proj_diff(length_xy=length_xy, display=False)
    shifts = Backend.to_numpy(model.shift_vector)
    assert shifts[0] == approx(-1, abs=0.2)
    assert shifts[1] == approx(-5, abs=0.2)
    assert shifts[2] == approx(13, abs=0.2)
