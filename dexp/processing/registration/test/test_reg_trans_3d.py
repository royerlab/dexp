from pytest import approx

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.demo.demo_reg_trans_3d import _register_translation_3d


def test_register_translation_3d_numpy():
    backend = NumpyBackend()
    register_translation_3d(backend)


def test_register_translation_3d_cupy():
    try:
        backend = CupyBackend()
        register_translation_3d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_translation_3d(backend, length_xy=128):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    image, shifted, unshifted, model = _register_translation_3d(backend, length_xy=length_xy, display=False)
    shifts = model.shift_vector
    assert shifts[0] == approx(-1, abs=0.2)
    assert shifts[1] == approx(-5, abs=0.2)
    assert shifts[2] == approx(13, abs=0.2)
