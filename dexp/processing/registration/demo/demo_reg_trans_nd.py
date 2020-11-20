from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.reg_trans_nd import register_translation_nd
from dexp.processing.registration.reg_trans_nd_maxproj import register_translation_maxproj_nd
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.utils.timeit import timeit


def demo_register_translation_nD_numpy():
    backend = NumpyBackend()
    register_translation_nD(backend)


def demo_register_translation_nD_cupy():
    try:
        backend = CupyBackend()
        register_translation_nD(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def register_translation_nD(backend, length_xy=320):
    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(backend,
                                                                                       add_noise=False,
                                                                                       shift=(1, 5, -13),
                                                                                       volume_fraction=0.5,
                                                                                       length_xy=length_xy,
                                                                                       length_z_factor=2)

    with timeit("register_translation_nd"):
        shifts, error = register_translation_nd(backend, image1, image2).get_shift_and_error()
        print(f"shifts: {shifts}, error: {error}")

    with timeit("register_translation_maxproj_nd"):
        shifts_maxproj, error_maxproj = register_translation_maxproj_nd(backend, image1, image2).get_shift_and_error()
        print(f"shifts: {shifts_maxproj}, error: {error_maxproj}")

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


demo_register_translation_nD_cupy()
demo_register_translation_nD_numpy()
