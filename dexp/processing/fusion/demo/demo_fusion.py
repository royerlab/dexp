import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.fusion.dct_fusion import fuse_dct_nd
from dexp.processing.fusion.dft_fusion import fuse_dft_nd
from dexp.processing.fusion.tg_fusion import fuse_tg_nd
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.utils.timeit import timeit


def demo_fusion_numpy():
    backend = NumpyBackend()
    demo_fusion(backend)


def demo_fusion_cupy():
    try:
        backend = CupyBackend()
        demo_fusion(backend, include_dct=False, length_xy=512)
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_fusion(backend, include_dct=True, length_xy=120):
    with timeit("generate data"):
        image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(backend, add_noise=True, length_xy=length_xy, length_z_factor=4)
        image_gt = backend.to_numpy(image_gt)

    with timeit("dct fusion"):
        image_fused_dct = fuse_dct_nd(backend, image1, image2) if include_dct else numpy.zeros_like(image_gt)
        image_fused_dct = backend.to_numpy(image_fused_dct)

    error_dct = numpy.median(numpy.abs(image_gt - image_fused_dct))
    print(f"error_dct={error_dct}")

    with timeit("dft fusion"):
        image_fused_dft = fuse_dft_nd(backend, image1, image2)
        image_fused_dft = backend.to_numpy(image_fused_dft)
    error_dft = numpy.median(numpy.abs(image_gt - image_fused_dft))
    print(f"error_dft={error_dft}")

    with timeit("tg fusion"):
        image_fused_tg = fuse_tg_nd(backend, image1, image2)
        image_fused_tg = backend.to_numpy(image_fused_tg)
    error_tg = numpy.median(numpy.abs(image_gt - image_fused_tg))
    print(f"error_tg={error_tg}")

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
        viewer.add_image(_c(image_fused_dct), name='image_fused_dct')
        viewer.add_image(_c(image_fused_dft), name='image_fused_dft')
        viewer.add_image(_c(image_fused_tg), name='image_fused_tg')


#demo_fusion_cupy()
demo_fusion_numpy()
