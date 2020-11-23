import pytest

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.kernels import gaussian_kernel_nd


def test_gaussian_numpy():
    backend = NumpyBackend()
    _test_gaussian(backend)


def test_gaussian_cupy():
    try:
        backend = CupyBackend()
        _test_gaussian(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_gaussian(backend):
    kernel = gaussian_kernel_nd(backend, ndim=3, size=17, sigma=2.0)

    assert backend.to_numpy(kernel.sum()) == pytest.approx(1)
