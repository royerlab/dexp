import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.processing.utils.element_wise_affine import element_wise_affine


def test_element_wise_affine_numpy():
    with NumpyBackend():
        _test_element_wise_affine()


def test_element_wise_affine_cupy():
    try:
        with CupyBackend():
            _test_element_wise_affine()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_element_wise_affine(length_xy=128):
    xp = Backend.get_xp_module()

    _, _, _, _, image, _ = generate_fusion_test_data(add_noise=False,
                                                     length_xy=length_xy,
                                                     length_z_factor=4)

    transformed = element_wise_affine(image, 2, 0.3)

    transformed = Backend.to_numpy(transformed)
    image = Backend.to_numpy(image)
    error = numpy.median(numpy.abs(image * 2 + 0.3 - transformed))
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
