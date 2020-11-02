from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.utils.timeit import timeit


def demo_multiview_data_numpy():
    backend = NumpyBackend()
    demo_multiview_data(backend)

def demo_multiview_data_cupy():
    try:
        backend = CupyBackend()
        demo_multiview_data(backend)
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")



def demo_multiview_data(backend, length_xy=320):
    with timeit("generate data"):
        image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(backend, add_noise=True, length_xy=length_xy, length_z_factor=4)


    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)
        viewer = Viewer()
        viewer.add_image(_c(image_gt), name='image_gt')
        viewer.add_image(_c(image_lowq), name='image_lowq')
        viewer.add_image(_c(blend_a), name='blend_a')
        viewer.add_image(_c(blend_b), name='blend_b')
        viewer.add_image(_c(image1), name='image1')
        viewer.add_image(_c(image2), name='image2')


demo_multiview_data_cupy()
demo_multiview_data_numpy()


