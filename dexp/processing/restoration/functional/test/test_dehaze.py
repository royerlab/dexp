from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.restoration.functional.dehazing import dehaze

from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def test_dehaze_numpy():
    backend = NumpyBackend()
    _test_dehaze(backend)

def test_dehaze_cupy():
    try:
        backend = CupyBackend()
        _test_dehaze(backend)
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def _test_dehaze(backend, length_xy=128):
    xp = backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(backend,
                                                                      add_noise=True,
                                                                      length_xy=length_xy,
                                                                      length_z_factor=4,
                                                                      independent_haze=True)

    with timeit('dehaze_new'):
        dehazed = dehaze(backend, image, size=25)
        dehazed = backend.to_numpy(dehazed)

    # compute [ercentage of haze removed:
    background_voxels_image = (1 - image_gt) * image
    background_voxels_dehazed = (1 - image_gt) * dehazed
    total_haze = xp.sum(background_voxels_image)
    total_remaining_haze = xp.sum(background_voxels_dehazed)
    percent_removed = (total_haze - total_remaining_haze) / total_haze
    print(f"percent_removed = {percent_removed}")

    # compute error on non-background voxels
    non_background_voxels_image = image_gt * image
    non_background_voxels_dehazed = image_gt * dehazed
    average_error = xp.mean(xp.absolute(non_background_voxels_image-non_background_voxels_dehazed))
    print(f"average_error = {average_error}")

    assert percent_removed>0.92
    assert percent_removed<12



