from skimage.registration import phase_cross_correlation

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.reg_trans_nd import register_translation_nd


def register_translation_2d_skimage(backend: Backend, image_a, image_b, upsample_factor: int = 16, **kwargs) -> TranslationRegistrationModel:
    image_a = backend.to_numpy(image_a)
    image_b = backend.to_numpy(image_b)
    shifts, error, _ = phase_cross_correlation(image_a, image_b, upsample_factor=upsample_factor, **kwargs)
    return TranslationRegistrationModel(shift_vector=shifts, error=error)


def register_translation_2d_dexp(backend: Backend, image_a, image_b, *args):
    return register_translation_nd(backend, image_a, image_b, *args)
