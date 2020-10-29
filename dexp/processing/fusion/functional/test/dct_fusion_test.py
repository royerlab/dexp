import numpy

from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.fusion.functional.dct_fusion import fuse_dct_nd
from dexp.processing.fusion.functional.test.fusion_test_data import generate_fusion_test_data


def test_dct_fusion_numpy():
    backend = NumpyBackend()
    dct_fusion(backend)


def dct_fusion(backend):
    image_gt, image_lowq, blend, image1, image2 = generate_fusion_test_data(add_noise=False)
    image_fused = fuse_dct_nd(backend, image1, image2)
    error = numpy.median(numpy.abs(image_gt - image_fused))
    print(error)
    assert error < 3

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image_gt, name='image_gt')
    #     viewer.add_image(image_lowq, name='image_lowq')
    #     viewer.add_image(image1, name='image1')
    #     viewer.add_image(image2, name='image2')
    #     viewer.add_image(image_fused, name='image_fused')


