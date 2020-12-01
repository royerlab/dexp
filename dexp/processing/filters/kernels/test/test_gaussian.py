import pytest

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.kernels.gaussian import gaussian_kernel_nd


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
    kernel = gaussian_kernel_nd(ndim=3, size=17, sigma=2.0)

    assert Backend.to_numpy(kernel.sum()) == pytest.approx(1)
