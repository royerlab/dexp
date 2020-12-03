from typing import Union, Tuple

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.warp_registration_model import WarpRegistrationModel
from dexp.processing.registration.reg_warp_nd import register_warp_nd
from dexp.utils.timeit import timeit


def register_warp_multiscale_nd(image_a,
                                image_b,
                                num_iterations: int = 3,
                                confidence_threshold: float = 0.5,
                                margin_ratios: Union[float, Tuple[float, ...]] = 0.2,
                                min_chunk: int = 32,
                                min_margin: int = 4,
                                **kwargs) -> WarpRegistrationModel:
    """
    Registers image_b to image_a using a multi-scale warp approach.


    Parameters
    ----------
    image_a : First image to register
    image_b : Second image to register
    num_iterations : Number of iterations: each iteration subdivides chunks by a factor 2.
    confidence_threshold : confidence threshold for chunk registration
    all additional kwargs are passed to register_warp_nd

    Returns
    -------
    WarpRegistrationModel

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if image_a.shape != image_b.shape:
        raise ValueError("Image must have same shape!")

    ndim = image_a.ndim

    if type(margin_ratios) is not tuple:
        margin_ratios = (margin_ratios,) * ndim

    vector_field = None
    confidence = None

    for i in range(num_iterations):

        # Clear memory allocation cache:
        Backend.current().clear_allocation_pool()

        # pre-apply transform from previous iterations:
        if vector_field is not None:
            model = WarpRegistrationModel(vector_field)
            _, image = model.apply(image_a, image_b)
        else:
            image = image_b

        with timeit(f"register_warp_nd {i}"):
            nb_div = 2 ** i
            # Note: the trick below is a 'ceil division' ceil(u/v) == -(-u//v)
            chunks = tuple(max(min_chunk, -(-s // nb_div)) for s in image_a.shape)
            margins = tuple(max(min_margin, int(c * r)) for c, r in zip(chunks, margin_ratios))
            # print(f"register iteration: {i}, div= {nb_div}, chunks={chunks}, margins={margins}")
            model = register_warp_nd(image_a, image, chunks=chunks, margins=margins, **kwargs)
            # print(f"mean confidence: {model.mean_confidence()}")
            # print(f"median shift magnitude: {model.median_shift_magnitude()}")
            model.clean(confidence_threshold=confidence_threshold)

            model_vector_field = Backend.to_backend(model.vector_field)
            model_confidence = Backend.to_backend(model.confidence)

        if vector_field is None:
            splits = tuple(s // max(min_chunk, -(-s // (2 ** (num_iterations - 1)))) for s in image_a.shape)
            # print(f"final resolution = {splits}")
            vector_field = xp.zeros(shape=splits + (ndim,), dtype=model_vector_field.dtype)
            confidence = xp.zeros(shape=splits, dtype=model_confidence.dtype)

        if model.mean_confidence() > confidence_threshold // 2:
            scale_factors = tuple(s / ms for ms, s in zip(model_confidence.shape, confidence.shape))
            # print(f"scale_factors: {scale_factors}")

            scaled_vector_field = sp.ndimage.zoom(model_vector_field, zoom=scale_factors + (1,), order=1)
            vector_field += scaled_vector_field

            scaled_confidence = sp.ndimage.zoom(model_confidence, zoom=scale_factors, order=1)
            confidence = xp.maximum(confidence, scaled_confidence)
        else:
            # print(f"Scale ignored!")
            break

    model = WarpRegistrationModel(vector_field=vector_field,
                                  confidence=confidence)

    return model
