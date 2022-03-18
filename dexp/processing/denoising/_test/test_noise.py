# flake8: noqa
from dexp.processing.denoising.demo.demo_noise import _demo_noise
from dexp.utils.backends import CupyBackend, NumpyBackend


def test_noise_numpy():
    with NumpyBackend():
        _test_noise()


def test_noise_cupy():
    try:
        with CupyBackend():
            _test_noise()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_noise():
    _demo_noise(display=False)
