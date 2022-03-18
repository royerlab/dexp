from dexp.processing.denoising.demo.demo_2D_butterworth import _demo_butterworth
from dexp.utils.backends import CupyBackend, NumpyBackend


def test_butterworth_numpy():
    with NumpyBackend():
        _test_butterworth()


def test_butterworth_cupy():
    try:
        with CupyBackend():
            _test_butterworth()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_butterworth():
    assert _demo_butterworth(display=False) >= 0.608 - 0.03
