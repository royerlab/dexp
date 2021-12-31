from skimage.registration import phase_cross_correlation

from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.processing.registration.translation_nd import register_translation_nd
from dexp.utils import xpArray
from dexp.utils.backends import Backend


def register_translation_2d_skimage(
    image_a: xpArray,
    image_b: xpArray,
    upsample_factor: int = 16,
    force_numpy: bool = False,
    internal_dtype=None,
    **kwargs
) -> TranslationRegistrationModel:
    """
    Registers two 2D images using just a translation-only model using skimage registration code.
    Warning: Works well unless images have lots of noise or overlap is partial.

    Parameters
    ----------
    image_a : First image
    image_b : Second Image
    upsample_factor : Upsampling factor for sub-pixel accuracy
    force_numpy : Forces output model to be allocated with numpy arrays.
    internal_dtype : internal dtype for computation
    kwargs : additional optional parameters

    Returns
    -------
    Translation-only registration model

    """
    image_a = Backend.to_numpy(image_a, dtype=internal_dtype)
    image_b = Backend.to_numpy(image_b, dtype=internal_dtype)
    shifts, error, _ = phase_cross_correlation(image_a, image_b, upsample_factor=upsample_factor, **kwargs)
    return TranslationRegistrationModel(shift_vector=shifts, confidence=error, force_numpy=force_numpy)


def register_translation_2d_dexp(image_a, image_b, **kwargs) -> TranslationRegistrationModel:
    """
    Registers two 2D images using just a translation-only model using dexp own registration code.

    Parameters
    ----------
    image_a : First image
    image_b : Second Image
    kwargs : additional optional parameters

    Returns
    -------
    Translation-only registration model

    """
    return register_translation_nd(image_a, image_b, **kwargs)
