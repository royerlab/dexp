import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.processing.utils.blend_images import blend_images


def test_blend_numpy():
    backend = NumpyBackend()
    _test_blend(backend)


def test_blend_cupy():
    try:
        backend = CupyBackend()
        _test_blend(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_blend(backend, length_xy=128):
    xp = backend.get_xp_module()

    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(backend,
                                                                                       add_noise=False,
                                                                                       length_xy=length_xy,
                                                                                       length_z_factor=4,
                                                                                       dtype=numpy.float32)

    blended = blend_images(backend, image1, image2, blend_a)

    assert blended is not image1
    assert blended is not image2
    assert blended.shape == image1.shape
    assert blended.shape == image2.shape
    assert blended.shape == blend_a.shape

    image_gt = backend.to_numpy(image_gt)
    blended = backend.to_numpy(blended)
    error = numpy.median(numpy.abs(image_gt - blended))
    print(f"error={error}")
    assert error < 22

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image_gt, name='image_gt')
    #     viewer.add_image(image1, name='image1')
    #     viewer.add_image(image2, name='image2')
    #     viewer.add_image(blend_a, name='blend_a')
    #     viewer.add_image(blended, name='blended')
