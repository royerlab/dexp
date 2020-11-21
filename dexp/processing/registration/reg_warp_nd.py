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

    def f(x, y):
        model = registration_method(backend, x, y, **kwargs)
        shift, error = model.get_shift_and_error()
        print(f"shift found: {shift}")
        return xp.asarray(shift), xp.asarray(error)

    vector_field, confidence = scatter_gather_i2v(backend,
                                                  f,
                                                  (image_a, image_b),
                                                  chunks=chunks,
                                                  margins=margins)

    return WarpRegistrationModel(vector_field=vector_field, confidence=confidence)
