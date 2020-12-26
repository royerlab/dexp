import pytest
from skimage.data import binary_blobs
from skimage.filters import gaussian
from skimage.util import random_noise

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.equalise.equalise_intensity import equalise_intensity


def test_equalise_intensity_numpy():
    with NumpyBackend():
        _equalise_intensity()


def test_equalise_intensity_cupy():
    try:
        with CupyBackend():
            _equalise_intensity()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _equalise_intensity(length=128):
    ratio_gt = 2.77

    image_1 = 300 * binary_blobs(length=length, n_dim=3, blob_size_fraction=0.04, volume_fraction=0.01).astype('f4')
    image_1 = gaussian(image_1, sigma=1)
    image_2 = image_1.copy() * ratio_gt

    image_1 = 95 + random_noise(image_1, mode='gaussian', var=0.5, clip=False)
    image_2 = 95 + random_noise(image_2, mode='gaussian', var=0.5, clip=False)

    # from napari import Viewer
    # with napari.gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image_1, name='image_1')
    #     viewer.add_image(image_2, name='image_2')

    org_image_1, org_image_2 = image_1.copy(), image_2.copy()
    equ_image_1, equ_image_2, corr_ratio = equalise_intensity(image_1, image_2)

    corr_ratio = Backend.to_numpy(corr_ratio)

    print(f" Ratio:{1 / corr_ratio}")

    assert ratio_gt == pytest.approx(1 / corr_ratio, 1e-2)

    # from napari import Viewer
    # with napari.gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(org_image_1, name='org_image_1', contrast_limits=(0,700))
    #     viewer.add_image(org_image_2, name='org_image_2', contrast_limits=(0,700))
    #     viewer.add_image(equ_image_1, name='equ_image_1', contrast_limits=(0,700))
    #     viewer.add_image(equ_image_2, name='equ_image_2', contrast_limits=(0,700))
