from pytest import approx

from dexp.processing.registration.demo.demo_translation_3d import (
    _register_translation_3d,
)
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def test_register_translation_3d_numpy():
    with NumpyBackend():
        register_translation_3d()


def test_register_translation_3d_cupy():
    try:
        with CupyBackend():
            register_translation_3d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_translation_3d(length_xy=128):
    _, _, _, model = _register_translation_3d(length_xy=length_xy, display=False)
    shifts = Backend.to_numpy(model.shift_vector)
    assert shifts[0] == approx(-1, abs=0.2)
    assert shifts[1] == approx(-5, abs=0.2)
    assert shifts[2] == approx(13, abs=0.2)
