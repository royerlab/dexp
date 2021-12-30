from dexp.processing.backends import CupyBackend, NumpyBackend
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
