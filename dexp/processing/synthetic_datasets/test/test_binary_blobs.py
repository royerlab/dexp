import numpy
import pytest
from skimage.data import binary_blobs as binary_blobs_skimage

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.binary_blobs import binary_blobs as binary_blobs_dexp


def test_blobs_numpy():
    backend = NumpyBackend()
    _test_blobs(backend)


def test_blobs_cupy():
    try:
        backend = CupyBackend()
        _test_blobs(backend)
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def _test_blobs(backend, length_xy=128):
    image_blobs = binary_blobs_dexp(backend, length=length_xy, n_dim=3, blob_size_fraction=0.07, volume_fraction=0.1).astype('f4')
    image_blobs = backend.to_numpy(image_blobs)

    image_blobs_skimage = binary_blobs_skimage(length=length_xy, n_dim=3, blob_size_fraction=0.07, volume_fraction=0.1).astype('f4')

    error = numpy.median(numpy.abs(image_blobs - image_blobs_skimage))
    print(error)
    assert error == pytest.approx(0)
