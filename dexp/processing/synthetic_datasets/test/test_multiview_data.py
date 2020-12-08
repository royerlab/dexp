from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.utils.timeit import timeit


def test_multiview_data_numpy():
    with NumpyBackend():
        _test_multiview_data()


def test_multiview_data_cupy():
    try:
        with CupyBackend():
            _test_multiview_data()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def _test_multiview_data(length_xy=128):
    with timeit("generate data"):
        image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(add_noise=False, length_xy=length_xy, length_z_factor=2)

    assert image_gt.shape == image_lowq.shape
    assert image_gt.shape == image1.shape
    assert image_gt.shape == image2.shape

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


test_multiview_data_cupy()
test_multiview_data_numpy()
