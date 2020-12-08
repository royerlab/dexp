import numpy
from numpy.linalg import norm
from skimage.data import camera
from skimage.util import random_noise

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.restoration.lipshitz_correction import lipschitz_continuity_correction
from dexp.utils.timeit import timeit


def test_lipschitz_continuity_correction_numpy():
    with NumpyBackend():
        _test_lipschitz_continuity_correction()


def test_lipschitz_continuity_correction_cupy():
    try:
        with CupyBackend():
            _test_lipschitz_continuity_correction()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def _test_lipschitz_continuity_correction(length_xy=128):
    image = camera().astype(numpy.float32) / 255
    noisy = random_noise(image.copy(), mode="gaussian", var=0.005, seed=0, clip=False)
    noisy = random_noise(noisy, mode="s&p", amount=0.03, seed=0, clip=False)

    with timeit('lipschitz_continuity_correction'):
        corrected = lipschitz_continuity_correction(image, lipschitz=0.15, in_place=False)
        corrected = Backend.to_numpy(corrected)

    assert corrected is not image
    assert corrected.shape == image.shape
    assert corrected.dtype == image.dtype

    error0 = norm(noisy - image, ord=1) / image.size
    error = norm(corrected - image, ord=1) / image.size

    print(f"Error noisy = {error0}")
    print(f"Error = {error}")

    assert error < 0.001
    assert error0 > 4 * error
