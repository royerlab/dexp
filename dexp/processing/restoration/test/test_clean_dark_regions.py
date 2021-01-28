import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def test_clean_dark_regions_numpy():
    with NumpyBackend():
        _test_clean_dark_regions()


def test_clean_dark_regions_cupy():
    try:
        with CupyBackend():
            _test_clean_dark_regions()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def _test_clean_dark_regions(length_xy=128):
    xp = Backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(add_noise=True,
                                                                      length_xy=length_xy,
                                                                      length_z_factor=1,
                                                                      independent_haze=False,
                                                                      background_stength=0.05,
                                                                      dtype=numpy.float32)

    # remove zero level
    image = xp.clip(image - 95, 0, None)

    with timeit('clean_dark_regions'):
        cleaned = clean_dark_regions(image, size=3, threshold=30, in_place=False)

    assert cleaned is not image
    assert cleaned.shape == image.shape
    assert cleaned.dtype == image.dtype

    # compute percentage of haze removed:
    background_voxels_image = (1 - image_gt) * image
    background_voxels_dehazed = (1 - image_gt) * cleaned
    total_haze = xp.sum(background_voxels_image)
    total_remaining_haze = xp.sum(background_voxels_dehazed)
    percent_removed = (total_haze - total_remaining_haze) / total_haze
    print(f"percent_removed = {percent_removed}")

    # compute error on non-background voxels
    non_background_voxels_image = image_gt * image
    non_background_voxels_dehazed = image_gt * cleaned
    average_error = xp.mean(xp.absolute(non_background_voxels_image - non_background_voxels_dehazed))
    print(f"average_error = {average_error}")

    assert percent_removed > 0.1
    assert percent_removed < 3000
