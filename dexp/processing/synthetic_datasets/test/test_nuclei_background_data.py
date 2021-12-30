from dexp.processing.backends import CupyBackend, NumpyBackend
from dexp.processing.synthetic_datasets.nuclei_background_data import (
    generate_nuclei_background_data,
)
from dexp.utils.timeit import timeit


def test_nuclei_background_data_numpy():
    with NumpyBackend():
        _test_nuclei_background_data()


def test_nuclei_background_data_cupy():
    try:
        with CupyBackend():
            _test_nuclei_background_data()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def _test_nuclei_background_data(length_xy=128):
    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(
            add_noise=True, length_xy=length_xy, length_z_factor=4
        )
    assert image_gt.shape == background.shape
    assert image_gt.shape == image.shape

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image_gt), name='image_gt')
    #     viewer.add_image(_c(image_lowq), name='image_lowq')
    #     viewer.add_image(_c(blend_a), name='blend_a')
    #     viewer.add_image(_c(blend_b), name='blend_b')
    #     viewer.add_image(_c(image1), name='image1')
    #     viewer.add_image(_c(image2), name='image2')


test_nuclei_background_data_cupy()
test_nuclei_background_data_numpy()
