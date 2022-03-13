# flake8: noqa
from dexp.processing.denoising.demo.demo_noise import _demo_noise
from dexp.utils.backends import NumpyBackend, CupyBackend


def test_noise_numpy():
    with NumpyBackend():
        _demo_noise(display=False)


def test_noise_cupy():
    try:
        with CupyBackend():
            _demo_noise(display=False)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")
