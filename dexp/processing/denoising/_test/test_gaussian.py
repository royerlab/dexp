from dexp.processing.denoising.demo.demo_2D_gaussian import _demo_gaussian
from dexp.utils.backends import CupyBackend, NumpyBackend


def test_gaussian_numpy():
    with NumpyBackend():
        _test_gaussian()


def test_gaussian_cupy():
    try:
        with CupyBackend():
            _test_gaussian()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_gaussian():
    assert _demo_gaussian(display=False) >= 0.600 - 0.02
