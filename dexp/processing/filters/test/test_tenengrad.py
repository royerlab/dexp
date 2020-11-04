from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.tenengrad import tenengrad
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data


def test_tenengrad_numpy():
    backend = NumpyBackend()
    _tenengrad(backend)


def test_tenengrad_cupy():
    try:
        backend = CupyBackend()
        _tenengrad(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _tenengrad(backend):
    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(backend, add_noise=False)

    tenengrad_image1 = tenengrad(backend, image1)
    tenengrad_image2 = tenengrad(backend, image2)

    from napari import Viewer, gui_qt
    with gui_qt():
        viewer = Viewer()
        viewer.add_image(image_gt, name='image_gt')
        viewer.add_image(image_lowq, name='image_lowq')
        viewer.add_image(blend_a, name='blend_a')
        viewer.add_image(blend_b, name='blend_b')
        viewer.add_image(image1, name='image1')
        viewer.add_image(image2, name='image2')
        viewer.add_image(tenengrad_image1, name='tenengrad_image1')
        viewer.add_image(tenengrad_image2, name='tenengrad_image2')
