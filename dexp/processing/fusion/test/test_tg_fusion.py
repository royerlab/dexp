import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.fusion.tg_fusion import fuse_tg_nd
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.utils.timeit import timeit


def test_tg_fusion_numpy():
    with NumpyBackend():
        tg_fusion()


def test_tg_fusion_cupy():
    try:
        with CupyBackend():
            tg_fusion()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def tg_fusion(length_xy=128):
    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(add_noise=False,
                                                                                       length_xy=length_xy,
                                                                                       length_z_factor=4)
    with timeit("dcf fusion + data transfer"):
        image_fused = fuse_tg_nd(image1, image2)
        image_fused = Backend.to_numpy(image_fused)

    image_gt = Backend.to_numpy(image_gt)
    error = numpy.median(numpy.abs(image_gt - image_fused))
    print(f"error={error}")
    assert error < 23

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image_gt, name='image_gt')
    #     viewer.add_image(image_lowq, name='image_lowq')
    #     viewer.add_image(blend_a, name='blend_a')
    #     viewer.add_image(blend_b, name='blend_b')
    #     viewer.add_image(image1, name='image1')
    #     viewer.add_image(image2, name='image2')
    #     viewer.add_image(image_fused, name='image_fused')
