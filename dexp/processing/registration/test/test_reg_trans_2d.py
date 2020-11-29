from pytest import approx

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.demo.demo_reg_trans_2d import _register_translation_2d


def test_register_translation_2d_numpy():
    backend = NumpyBackend()
    register_translation_2d(backend)


def test_register_translation_2d_cupy():
    try:
        backend = CupyBackend()
        register_translation_2d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_translation_2d(backend):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    image, shifted, unshifted, model = _register_translation_2d(backend, display=False)

    shifts = model.shift_vector
    assert shifts[0] == approx(-13, abs=0.5)
    assert shifts[1] == approx(5, abs=0.5)

    error_shifted = xp.mean(xp.absolute(image - shifted))
    error_unshifted = xp.mean(xp.absolute(image - unshifted))
    print(f"error_shifted = {error_shifted}")
    print(f"error_unshifted = {error_unshifted}")

    assert error_unshifted < error_shifted
    assert error_unshifted < 0.02
