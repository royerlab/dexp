import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.synthetic_datasets.nuclei_background_data import (
    generate_nuclei_background_data,
)
from dexp.utils.timeit import timeit


def test_warp_3d_cupy():
    try:
        with CupyBackend():
            _test_warp_3d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_warp_3d(length_xy=256, grid_size=8):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    with timeit("generate data"):
        _, _, image = generate_nuclei_background_data(
            add_noise=True, length_xy=length_xy, length_z_factor=1, zoom=2, dtype=numpy.float32
        )

    newimage = image[0:512, 0:511, 0:509]
    image = newimage
    image = Backend.to_backend(image)

    print(f"shape={image.shape}")

    vector_field = numpy.random.uniform(low=-5, high=+5, size=(grid_size,) * 3 + (3,))

    with timeit("warp"):
        warped = warp(image, vector_field, vector_field_upsampling=4)

    with timeit("dewarp"):
        dewarped = warp(warped, -vector_field, vector_field_upsampling=4)

    error = xp.mean(xp.absolute(image - dewarped))
    print(f"error = {error}")

    assert error < 40

    shifted_ndimage = sp.ndimage.shift(image, shift=(11, 5, -17))
    vector_field = xp.asarray([11, 5, -17])[numpy.newaxis, numpy.newaxis, numpy.newaxis]
    shifted_warp = warp(image, vector_field)

    error_ndimage_warp = xp.mean(xp.absolute(shifted_ndimage - shifted_warp))
    print(f"error_ndimage_warp = {error_ndimage_warp}")
    assert error_ndimage_warp < 1e-3
