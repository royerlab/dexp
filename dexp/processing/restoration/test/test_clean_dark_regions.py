from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions

from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def test_clean_dark_regions_numpy():
    backend = NumpyBackend()
    _test_clean_dark_regions(backend)


def test_clean_dark_regions_cupy():
    try:
        backend = CupyBackend()
        _test_clean_dark_regions(backend)
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def _test_clean_dark_regions(backend, length_xy=128):
    xp = backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(backend,
                                                                      add_noise=True,
                                                                      length_xy=length_xy,
                                                                      length_z_factor=4,
                                                                      independent_haze=False,
                                                                      background_stength=0.05)

    # remove zero level
    image -= 95
    image_gt -= 95

    with timeit('clean_dark_regions'):
        cleaned = clean_dark_regions(backend, image, size=3, threshold=100)

    assert cleaned is not image
    assert cleaned.shape == image.shape
    assert cleaned.dtype == image.dtype

    # compute [ercentage of haze removed:
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