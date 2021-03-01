from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.color.demo.demo_scale_bar import demo_scale_bar


def test_scale_bar_numpy():
    with NumpyBackend():
        _test_scale_bar()


def test_scale_bar_cupy():
    try:
        with CupyBackend():
            _test_scale_bar()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_scale_bar():
    demo_scale_bar(display=False)
