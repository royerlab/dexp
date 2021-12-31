from dexp.processing.color.demo.demo_border import demo_border
from dexp.utils.backends import CupyBackend, NumpyBackend


def test_border_numpy():
    with NumpyBackend():
        _test_border()


def test_border_cupy():
    try:
        with CupyBackend():
            _test_border()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_border():
    demo_border(display=False)
