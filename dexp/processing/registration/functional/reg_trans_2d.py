from dexp.processing.backends.backend import Backend
from skimage.feature import register_translation

from dexp.processing.registration.functional.reg_trans_nd import register_translation_nd
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel


def register_translation_2d_skimage(backend: Backend, image_a, image_b, upsample_factor: int = 16, *args) -> TranslationRegistrationModel:
    image_a = backend.to_numpy(image_a)
    image_b = backend.to_numpy(image_b)
    shifts, error, _ = register_translation(image_a, image_b, upsample_factor=upsample_factor, *args)
    return TranslationRegistrationModel(shift_vector=shifts, error=error)

def register_translation_2d_dexp(backend:Backend, image_a, image_b, *args):
    return register_translation_nd(backend, image_a, image_b, *args)










