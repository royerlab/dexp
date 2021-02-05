from pytest import approx

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.demo.demo_translation_3d_proj import _register_translation_3d_proj
from dexp.processing.registration.translation_2d import register_translation_2d_skimage, register_translation_2d_dexp


def test_register_translation_3d_maxproj_numpy():
    with NumpyBackend():
        register_translation_3d_maxproj(register_translation_2d_skimage)
        register_translation_3d_maxproj(register_translation_2d_dexp)


def test_register_translation_3d_maxproj_cupy():
    try:
        with CupyBackend():
            register_translation_3d_maxproj(register_translation_2d_skimage)
            register_translation_3d_maxproj(register_translation_2d_dexp)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_translation_3d_maxproj(method, length_xy=128):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    image, shifted, unshifted, model = _register_translation_3d_proj(length_xy=length_xy, method=method, display=False)
    shifts = Backend.to_numpy(model.shift_vector)
    assert shifts[0] == approx(-1, abs=0.5)
    assert shifts[1] == approx(-5, abs=0.5)
    assert shifts[2] == approx(13, abs=0.5)
