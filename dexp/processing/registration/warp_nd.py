from typing import Tuple, Union

from arbol import aprint, section

from dexp.processing.registration.model.warp_registration_model import (
    WarpRegistrationModel,
)
from dexp.processing.registration.translation_nd_proj import (
    register_translation_proj_nd,
)
from dexp.processing.utils.scatter_gather_i2v import scatter_gather_i2v
from dexp.utils import xpArray
from dexp.utils.backends import Backend


@section("register_warp_nd")
def register_warp_nd(
    image_a: xpArray,
    image_b: xpArray,
    chunks: Union[int, Tuple[int, ...]],
    margins: Union[int, Tuple[int, ...]] = None,
    registration_method=register_translation_proj_nd,
    force_numpy: bool = False,
    **kwargs,
) -> WarpRegistrationModel:
    """
    Registers two nD images using warp model (piece-wise translation model).


    Parameters
    ----------
    image_a : First image to register
    image_b : Second image to register
    chunks : Chunk sizes to divide image into
    margins : Margins to add along each dimension per chunk
    registration_method : registration method to use per tile, must return a TranslationRegistrationModel.
    force_numpy: Forces output model to be allocated with numpy arrays.
    all additional kwargs are passed to the registration method (by default register_translation_maxproj_nd)

    Returns
    -------
    Warp registration model

    """
    image_a = Backend.to_backend(image_a)
    image_b = Backend.to_backend(image_b)

    xp = Backend.get_xp_module()

    def f(x, y):
        model = registration_method(x, y, force_numpy=force_numpy, **kwargs)
        aprint(f"model: {model} {'' if model.confidence > 0.3 else '(LOW QUALITY!)'}")
        shift, confidence = model.get_shift_and_confidence()
        return xp.asarray(shift), xp.asarray(confidence)

    vector_field, confidence = scatter_gather_i2v((image_a, image_b), f, tiles=chunks, margins=margins)

    model = WarpRegistrationModel(vector_field=vector_field, confidence=confidence, force_numpy=force_numpy)

    return model
