from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.reg_trans_2d import register_translation_2d_dexp
from dexp.processing.registration.reg_trans_nd_maxproj import register_translation_maxproj_nd
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data
from dexp.utils.timeit import timeit


def demo_register_translation_3d_maxproj_numpy():
    with NumpyBackend():
        _register_translation_3d_maxproj()


def demo_register_translation_3d_maxproj_cupy():
    try:
        with CupyBackend():
            _register_translation_3d_maxproj()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _register_translation_3d_maxproj(method=register_translation_2d_dexp, length_xy=256, display=True):
    with timeit("generate dataset"):
        image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(add_noise=False,
                                                                                           shift=(1, 5, -13),
                                                                                           volume_fraction=0.5,
                                                                                           length_xy=length_xy,
                                                                                           length_z_factor=1)

    with timeit("register_translation_maxproj_nd"):
        model = register_translation_maxproj_nd(image1, image2, register_translation_2d=method)
        print(f"model: {model}")

    with timeit("shift back"):
        image1_reg, image2_reg = model.apply(image1, image2, pad=False)
        image1_reg_pad, image2_reg_pad = model.apply(image1, image2, pad=True)

    if display:
        from napari import Viewer, gui_qt
        with gui_qt():
            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image_gt), name='image_gt', visible=False)
            viewer.add_image(_c(image_lowq), name='image_lowq', visible=False)
            viewer.add_image(_c(blend_a), name='blend_a', visible=False)
            viewer.add_image(_c(blend_b), name='blend_b', visible=False)
            viewer.add_image(_c(image1), name='image1', colormap='bop orange', blending='additive', visible=False)
            viewer.add_image(_c(image2), name='image2', colormap='bop blue', blending='additive', visible=False)
            viewer.add_image(_c(image1_reg), name='image1_reg', colormap='bop orange', blending='additive', visible=False)
            viewer.add_image(_c(image2_reg), name='image2_reg', colormap='bop blue', blending='additive', visible=False)
            viewer.add_image(_c(image1_reg_pad), name='image1_reg_pad', colormap='bop orange', blending='additive')
            viewer.add_image(_c(image2_reg_pad), name='image2_reg_pad', colormap='bop blue', blending='additive')

    return image1, image2, image2_reg, model


if __name__ == "__main__":
    demo_register_translation_3d_maxproj_cupy()
    demo_register_translation_3d_maxproj_numpy()
