from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.fusion.functional.dft_fusion import fuse_dft_nd
from dexp.processing.fusion.functional.test.fusion_test_data import generate_fusion_test_data


def test_dft_fusion_numpy():
    backend = NumpyBackend()
    dft_fusion_numpy(backend)


def dft_fusion_numpy(backend):
    image_gt, image_lowq, blend, image1, image2 = generate_fusion_test_data()

    fused = fuse_dft_nd(backend, image1, image2)

    from napari import Viewer, gui_qt
    with gui_qt():
        viewer = Viewer()
        viewer.add_image(image_gt, name='image_gt')
        viewer.add_image(image_lowq, name='image_lowq')
        viewer.add_image(image1, name='image1')
        viewer.add_image(image2, name='image2')
        viewer.add_image(fused, name='fused')


