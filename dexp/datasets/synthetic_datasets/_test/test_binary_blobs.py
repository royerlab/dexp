import numpy
import pytest
from skimage.data import binary_blobs as binary_blobs_skimage

from dexp.datasets.synthetic_datasets import binary_blobs as binary_blobs_dexp
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def test_blobs_numpy():
    with NumpyBackend():
        _test_blobs()


def test_blobs_cupy():
    try:
        with CupyBackend():
            _test_blobs()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def _test_blobs(length_xy=128):
    image_blobs = binary_blobs_dexp(length=length_xy, n_dim=3, blob_size_fraction=0.07, volume_fraction=0.1).astype(
        "f4"
    )
    image_blobs = Backend.to_numpy(image_blobs)

    image_blobs_skimage = binary_blobs_skimage(
        length=length_xy, n_dim=3, blob_size_fraction=0.07, volume_fraction=0.1
    ).astype("f4")

    error = numpy.median(numpy.abs(image_blobs - image_blobs_skimage))
    print(error)
    assert error == pytest.approx(0)
