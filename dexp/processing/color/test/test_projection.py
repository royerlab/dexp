from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.color.demo.demo_projection import demo_projection


def test_projection_numpy():
    with NumpyBackend():
        _test_projection()


def test_projection_cupy():
    try:
        with CupyBackend():
            _test_projection()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_projection():
    demo_projection(length_xy=64, display=False)
