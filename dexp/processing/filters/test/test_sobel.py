from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.sobel_filter import sobel_filter
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data


def test_sobel_numpy():
    with NumpyBackend():
        _sobel()


def test_sobel_cupy():
    try:
        with CupyBackend():
            _sobel()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _sobel(length_xy=128):
    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(length_xy=length_xy,
                                                                                       add_noise=False)

    tenengrad_image1 = sobel_filter(image1)
    tenengrad_image2 = sobel_filter(image2)

    assert tenengrad_image1.shape == image1.shape
    assert tenengrad_image2.shape == image2.shape
    assert tenengrad_image1.dtype == image2.dtype
    assert tenengrad_image2.dtype == image2.dtype

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image_gt, name='image_gt')
    #     viewer.add_image(image_lowq, name='image_lowq')
    #     viewer.add_image(blend_a, name='blend_a')
    #     viewer.add_image(blend_b, name='blend_b')
    #     viewer.add_image(image1, name='image1')
    #     viewer.add_image(image2, name='image2')
    #     viewer.add_image(tenengrad_image1, name='tenengrad_image1')
    #     viewer.add_image(tenengrad_image2, name='tenengrad_image2')
