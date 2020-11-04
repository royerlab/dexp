from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data


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

    model = TranslationRegistrationModel(shift_vector=(-1, -5, 13))

    image1_reg, image2_reg = model.apply(backend, image1, image2, pad=False)
    image1_reg_pad, image2_reg_pad = model.apply(backend, image1, image2, pad=True)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_gt), name='image_gt')
        viewer.add_image(_c(image1), name='image1')
        viewer.add_image(_c(image2), name='image2')
        viewer.add_image(_c(image1_reg), name='image1_reg')
        viewer.add_image(_c(image2_reg), name='image2_reg')
        viewer.add_image(_c(image1_reg_pad), name='image1_reg_pad')
        viewer.add_image(_c(image2_reg_pad), name='image2_reg_pad')


demo_register_translation_nD_cupy()
demo_register_translation_nD_numpy()
