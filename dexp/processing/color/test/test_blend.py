from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.color.demo.demo_blend import demo_blend


def test_blend_numpy():
    with NumpyBackend():
        _test_blend()


def test_blend_cupy():
    try:
        with CupyBackend():
            _test_blend()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_blend():
    demo_blend(display=False)
