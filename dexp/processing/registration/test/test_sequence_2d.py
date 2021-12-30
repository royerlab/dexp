from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.registration.demo.demo_sequence_2d import _register_sequence_2d


def test_register_translation_2d_cupy():
    try:
        with CupyBackend():
            register_sequence_2d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_sequence_2d(length_xy=256, n=128):
    image, _, _, model = _register_sequence_2d(length_xy=length_xy, n=n, display=False)

    assert len(model) == image.shape[0]
