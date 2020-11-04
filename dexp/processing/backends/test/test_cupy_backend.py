import numpy
import pytest
from skimage.data import binary_blobs

from dexp.processing.backends.cupy_backend import CupyBackend


def test_cupy_backend():
    try:
        array = binary_blobs(length=100, n_dim=3, blob_size_fraction=0.01, volume_fraction=0.01).astype(numpy.float32)

        backend = CupyBackend()

        array_b = backend.to_backend(array, numpy.float64)
        array_r = backend.to_numpy(array_b, numpy.float32)

        assert pytest.approx(array, rel=1e-5) == array_r
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")
