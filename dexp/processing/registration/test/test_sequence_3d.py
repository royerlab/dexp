from dexp.processing.backends import CupyBackend
from dexp.processing.registration.demo.demo_sequence_3d import _register_sequence_3d


def test_register_translation_3d_cupy():
    try:
        with CupyBackend():
            register_sequence_3d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_sequence_3d(length_xy=256, n=128):
    image, _, _, model = _register_sequence_3d(length_xy=length_xy, n=n, display=False)

    assert len(model) == image.shape[0]
