import numpy
import pytest
from skimage.data import binary_blobs

from dexp.processing.backends.pytorch_backend import PytorchBackend


def test_pytorch_backend():
    array = binary_blobs(length=100, n_dim=3, blob_size_fraction=0.01, volume_fraction=0.01).astype(numpy.float32)

    backend = PytorchBackend()

    array_b = backend.to_backend(array, numpy.float64)
    array_r = backend.to_numpy(array_b, numpy.float32)

    assert pytest.approx(array, rel=1e-5) == array_r