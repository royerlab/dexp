from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data


def test_translation_model_numpy():
    with NumpyBackend():
        _test_translation_model()


def test_translation_model_cupy():
    try:
        with CupyBackend():
            _test_translation_model()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _test_translation_model(length_xy=128):
    xp = Backend.get_xp_module()

    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(add_noise=False,
                                                                                       shift=(1, 5, -13),
                                                                                       volume_fraction=0.5,
                                                                                       length_xy=length_xy,
                                                                                       length_z_factor=2,
                                                                                       amount_low=0,
                                                                                       zero_level=0)

    model = TranslationRegistrationModel(shift_vector=(-1, -5, 13))

    image1_reg, image2_reg = model.apply_pair(image1, image2, pad=False)
    dumb_fusion = xp.maximum(image1_reg, image2_reg)
    average_error = xp.mean(xp.absolute(dumb_fusion - image_gt))
    print(f"average_error = {average_error}")

    image1_reg_pad, image2_reg_pad = model.apply_pair(image1, image2, pad=True)
    image1_reg_pad = image1_reg_pad[0:length_xy // 2, 0:length_xy, 0:length_xy]
    image2_reg_pad = image2_reg_pad[0:length_xy // 2, 0:length_xy, 0:length_xy]
    dumb_fusion_pad = xp.maximum(image1_reg_pad, image2_reg_pad)
    image_gt_shifted = xp.roll(image_gt, shift=(1, 5, 0), axis=(0, 1, 2))
    image_gt_shifted = image_gt_shifted[0:length_xy // 2, 0:length_xy, 0:length_xy]
    average_error_pad = xp.mean(xp.absolute(dumb_fusion_pad - image_gt_shifted))
    print(f"average_error_pad = {average_error_pad}")

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image1), name='image1')
    #     viewer.add_image(_c(image2), name='image2')
    #     viewer.add_image(_c(image1_reg), name='image1_reg')
    #     viewer.add_image(_c(image2_reg), name='image2_reg')
    #     viewer.add_image(_c(dumb_fusion), name='dumb_fusion')
    #     viewer.add_image(_c(image_gt), name='image_gt')
    #     viewer.add_image(_c(image1_reg_pad), name='image1_reg_pad')
    #     viewer.add_image(_c(image2_reg_pad), name='image2_reg_pad')
    #     viewer.add_image(_c(dumb_fusion_pad), name='dumb_fusion_pad')
    #     viewer.add_image(_c(image_gt_shifted), name='image_gt_shifted')

    assert average_error < 11
    assert average_error_pad < 11
