from dexp.datasets.synthetic_datasets import generate_fusion_test_data
from dexp.processing.filters.sobel_filter import sobel_filter
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_sobel_numpy():
    with NumpyBackend():
        _demo_sobel()


def demo_sobel_cupy():
    try:
        with CupyBackend():
            _demo_sobel()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_sobel():
    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(add_noise=False)

    tenengrad_image1 = sobel_filter(image1)
    tenengrad_image2 = sobel_filter(image2)

    assert tenengrad_image1.shape == image1.shape
    assert tenengrad_image2.shape == image2.shape
    assert tenengrad_image1.dtype == image2.dtype
    assert tenengrad_image2.dtype == image2.dtype

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_gt), name="image_gt")
        viewer.add_image(_c(image_lowq), name="image_lowq")
        viewer.add_image(_c(blend_a), name="blend_a")
        viewer.add_image(_c(blend_b), name="blend_b")
        viewer.add_image(_c(image1), name="image1")
        viewer.add_image(_c(image2), name="image2")
        viewer.add_image(_c(tenengrad_image1), name="tenengrad_image1")
        viewer.add_image(_c(tenengrad_image2), name="tenengrad_image2")


if __name__ == "__main__":
    demo_sobel_cupy()
    demo_sobel_numpy()
