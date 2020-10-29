import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.fusion.functional.dct_fusion import fuse_dct_nd
from dexp.processing.fusion.functional.dft_fusion import fuse_dft_nd
from dexp.processing.fusion.functional.test.fusion_test_data import generate_fusion_test_data

def demo_fusion_numpy():
    backend = NumpyBackend()
    demo_fusion(backend)

def demo_fusion_cupy():
    try:
        backend = CupyBackend()
        demo_fusion(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! ignored!")



def demo_fusion(backend):
    image_gt, image_lowq, blend, image1, image2 = generate_fusion_test_data(add_noise=True)
    image_fused_dct = fuse_dct_nd(backend, image1, image2)
    image_fused_dft = fuse_dft_nd(backend, image1, image2)
    error_dct = numpy.median(numpy.abs(image_gt - image_fused_dct))
    error_dft = numpy.median(numpy.abs(image_gt - image_fused_dft))
    print(f"error_dct={error_dct}")
    print(f"error_dft={error_dft}")

    from napari import Viewer, gui_qt
    with gui_qt():
        viewer = Viewer()
        viewer.add_image(image_gt, name='image_gt')
        viewer.add_image(image_lowq, name='image_lowq')
        viewer.add_image(image1, name='image1')
        viewer.add_image(image2, name='image2')
        viewer.add_image(image_fused_dct, name='image_fused_dct')
        viewer.add_image(image_fused_dft, name='image_fused_dft')


demo_fusion_numpy()
demo_fusion_cupy()

