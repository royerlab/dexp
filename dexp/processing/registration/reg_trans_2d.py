from skimage.registration import phase_cross_correlation

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.reg_trans_nd import register_translation_nd


def register_translation_2d_skimage(backend: Backend, image_a, image_b, upsample_factor: int = 16, **kwargs) -> TranslationRegistrationModel:
    """
    Registers two 2D images using just a translation-only model using skimage registration code.
    Warning: Works well unless images have lots of noise or overlap is partial.

    Parameters
    ----------
    backend : Backend to use
    image_a : First image
    image_b : Second Image
    upsample_factor : Upsampling factor for sub-pixel accuracy
    kwargs : additional optional parameters

    Returns
    -------
    Translation-only registration model

    """
    image_a = backend.to_numpy(image_a)
    image_b = backend.to_numpy(image_b)
    shifts, error, _ = phase_cross_correlation(image_a, image_b, upsample_factor=upsample_factor, **kwargs)
    return TranslationRegistrationModel(shift_vector=shifts, error=error)


def register_translation_2d_dexp(backend: Backend, image_a, image_b, *args) -> TranslationRegistrationModel:
    """
    Registers two 2D images using just a translation-only model using dexp own registration code.

    Parameters
    ----------
    backend : Backend to use
    image_a : First image
    image_b : Second Image
    upsample_factor : Upsampling factor for sub-pixel accuracy
    kwargs : additional optional parameters

    Returns
    -------
    Translation-only registration model

    """
    return register_translation_nd(backend, image_a, image_b, *args)
