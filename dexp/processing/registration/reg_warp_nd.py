from typing import Union, Tuple

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.warp_registration_model import WarpRegistrationModel
from dexp.processing.registration.reg_trans_nd_maxproj import register_translation_maxproj_nd
from dexp.processing.utils.scatter_gather_i2v import scatter_gather_i2v


def register_warp_nd(backend: Backend,
                     image_a,
                     image_b,
                     chunks: Union[int, Tuple[int, ...]],
                     margins: Union[int, Tuple[int, ...]] = None,
                     registration_method=register_translation_maxproj_nd,
                     **kwargs) -> WarpRegistrationModel:
    """
    Registers two nD images using warp model (piece-wise translation model).


    Parameters
    ----------
    backend : backend for computation
    image_a : First image to register
    image_b : Second image to register
    chunks : Chunk sizes to divide image into
    margins : Margins to add along each dimension per chunk
    registration_method : registration method to use per tile, must return a TranslationRegistrationModel.
    all additional kwargs are passed to the registration method (by default register_translation_maxproj_nd)

    Returns
    -------
    Warp registration model

    """
    image_a = backend.to_backend(image_a)
    image_b = backend.to_backend(image_b)

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    def f(x, y):
        model = registration_method(backend, x, y, **kwargs)
        # if model.confidence > 0.3:
        print(f"model: {model}")
        shift, confidence = model.get_shift_and_confidence()
        return xp.asarray(shift), xp.asarray(confidence)

    vector_field, confidence = scatter_gather_i2v(backend,
                                                  f,
                                                  (image_a, image_b),
                                                  chunks=chunks,
                                                  margins=margins)

    return WarpRegistrationModel(vector_field=vector_field, confidence=confidence)
