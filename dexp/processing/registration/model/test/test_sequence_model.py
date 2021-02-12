from arbol import aprint

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.model.sequence_registration_model import SequenceRegistrationModel
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.synthetic_datasets.multiview_data import generate_fusion_test_data


def test_sequence_model_numpy():
    with NumpyBackend():
        _test_sequence_model()


def test_sequence_model_cupy():
    try:
        with CupyBackend():
            _test_sequence_model()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _test_sequence_model(length_xy=128):
    xp = Backend.get_xp_module()

    model = SequenceRegistrationModel()

    assert len(model) == 0

    model_list = list(TranslationRegistrationModel(shift_vector=[1 * (1 + 0.2 * i), 5 + 0.5 * i, -13 - i], confidence=0.6) for i in range(0, 10))
    model = SequenceRegistrationModel(model_list)
    assert len(model) == 10

    image_gt, image_lowq, blend_a, blend_b, image1, image2 = generate_fusion_test_data(add_noise=False,
                                                                                       shift=(-1, -5, 13),
                                                                                       volume_fraction=0.5,
                                                                                       length_xy=length_xy,
                                                                                       length_z_factor=2,
                                                                                       amount_low=0,
                                                                                       zero_level=0)

    reg_images = list(model.apply(image2, index=index, pad=False) for index in range(0, 10))

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return Backend.to_numpy(array)
    #     viewer = Viewer()
    #     for i, reg_image in enumerate(reg_images):
    #         viewer.add_image(_c(reg_image), name=f'reg_image_{i}')

    average_errors = list(float(xp.mean(xp.absolute(reg_image - image1))) for reg_image in reg_images)
    aprint(average_errors)
    for u, e0 in enumerate(average_errors):
        for v, e1 in enumerate(average_errors):
            if u + 1 < v:
                assert e0 < e1

    reg_images_pad = list(model.apply(image2, index=index, pad=True) for index in range(0, 10))

    assert reg_images_pad[0].shape == model.padded_shape(image2.shape)
    assert model.padding() == ((0, 3), (0, 10), (22, 00))
