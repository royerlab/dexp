import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.processing.utils.normalise import normalise


def test_normalise_numpy():
    backend = NumpyBackend()
    _test_normalise(backend)


def test_normalise_cupy():
    try:
        backend = CupyBackend()
        _test_normalise(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_normalise(backend, length_xy=128):
    xp = backend.get_xp_module()

    _, _, image = generate_nuclei_background_data(backend, add_noise=True,
                                                  length_xy=length_xy,
                                                  length_z_factor=4)

    image = image.astype(numpy.float16)

    image_normalised, denorm_fun = normalise(backend, image, low=-0.5, high=1)
    image_denormalised = denorm_fun(image_normalised)

    assert image_normalised.dtype == image.dtype
    assert image_denormalised.dtype == image.dtype

    assert image_normalised.shape == image.shape
    assert image_denormalised.shape == image.shape

    assert image_normalised.min() >= -0.5
    assert image_normalised.max() <= 1
    assert image_normalised.max() - image_normalised.min() >= 1.5

    assert image_denormalised.min() >= image.min()
    assert image_denormalised.max() <= image.max()
    assert image_denormalised.max() - image_denormalised.min() >= image.max() - image.min()

    image = backend.to_numpy(image)
    image_denormalised = backend.to_numpy(image_denormalised)
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
