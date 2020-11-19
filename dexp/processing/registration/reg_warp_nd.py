import math
from typing import Union, Tuple

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.model.warp_registration_model import WarpRegistrationModel
from dexp.processing.utils.scatter_gather_i2v import scatter_gather_i2v


def register_warp_nd(backend: Backend,
                    image_a,
                    image_b,
                    chunks: Union[int, Tuple[int, ...]],
                    margins: Union[int, Tuple[int, ...]] = None,
                    max_range_ratio: float = 0.5,
                    fine_window_radius: int = 4,
                    decimate: int = 16,
                    quantile: float = 0.999,
                    internal_dtype=numpy.float32,
                    log: bool = False) -> TranslationRegistrationModel:
    """
    Registers two nD images using warp model (piece-wise translation model).


    Parameters
    ----------
    backend : backend for computation
    image_a : First image to register
    image_b : Second image to register
    max_range_ratio : backend for computation
    fine_window_radius : Window of which to refine translation estimate
    decimate : How much to decimate when computing floor level
    quantile : Quantile to use for robust min and max
    internal_dtype : internal dtype for computation
    log : output logging information, or not.

    Returns
    -------
    Warp registration model

    """
    image_a = backend.to_backend(image_a)
    image_b = backend.to_backend(image_b)

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    def f(x,y):
        #TODO: register x and y and return the shift vector
        return 0

    result = scatter_gather_i2v(backend,
                                f,
                                (image_a, image_b),
                                chunks=chunks,
                                margins=margins)

    #TODO: maybe give the possibility to return a tupple of arrays instead of a single array? so that confidence can be returned?
    vector_field=0
    confidence=0

    return WarpRegistrationModel(vector_field=vector_field, confidence=confidence)

