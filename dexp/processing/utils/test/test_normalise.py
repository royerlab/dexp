import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.processing.utils.normalise import normalise_functions


def test_normalise_numpy():
    with NumpyBackend():
        _test_normalise()


def test_normalise_cupy():
    try:
        with CupyBackend():
            _test_normalise()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_normalise(length_xy=128):
    xp = Backend.get_xp_module()

    _, _, image = generate_nuclei_background_data(add_noise=True,
                                                  length_xy=length_xy,
                                                  length_z_factor=4,
                                                  dtype=numpy.float32)

    image = image.astype(numpy.uint16)

    norm_fun, denorm_fun = normalise_functions(image, low=-0.5, high=1, in_place=False, clip=True, dtype=numpy.float32)

    image_normalised = norm_fun(image)
    image_denormalised = denorm_fun(image_normalised)

    assert image_normalised.dtype == numpy.float32
    assert image_denormalised.dtype == image.dtype

    assert image_normalised.shape == image.shape
    assert image_denormalised.shape == image.shape

    assert image_normalised.min() >= -0.5
    assert image_normalised.max() <= 1
    assert image_normalised.max() - image_normalised.min() >= 1.5

    assert image_denormalised.min() * (1 + 1e-3) >= image.min()
    assert image_denormalised.max() <= (1 + 1e-3) * image.max()
    assert (image_denormalised.max() - image_denormalised.min()) * (1 + 1e-3) >= image.max() - image.min()

    image = Backend.to_numpy(image)
    image_denormalised = Backend.to_numpy(image_denormalised)
    error = numpy.median(numpy.abs(image - image_denormalised))
    print(f"error={error}")
    assert error < 0.02

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image_gt, name='image_gt')
    #     viewer.add_image(image1, name='image1')
    #     viewer.add_image(image2, name='image2')
    #     viewer.add_image(blend_a, name='blend_a')
    #     viewer.add_image(blended, name='blended')
