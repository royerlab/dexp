from typing import Tuple, Union

from arbol import aprint, asection, section

from dexp.processing.registration.model.warp_registration_model import (
    WarpRegistrationModel,
)
from dexp.processing.registration.warp_nd import register_warp_nd
from dexp.utils import xpArray
from dexp.utils.backends import Backend


@section("register_warp_multiscale_nd")
def register_warp_multiscale_nd(
    image_a: xpArray,
    image_b: xpArray,
    num_iterations: int = 3,
    confidence_threshold: float = 0.5,
    max_residual_shift: float = None,
    margin_ratios: Union[float, Tuple[float, ...]] = 0.2,
    min_chunk: int = 32,
    min_margin: int = 4,
    save_memory: bool = False,
    force_numpy: bool = False,
    **kwargs,
) -> WarpRegistrationModel:
    """
    Registers image_b to image_a using a multi-scale warp approach.


    Parameters
    ----------
    image_a : First image to register
    image_b : Second image to register
    num_iterations : Number of iterations: each iteration subdivides chunks by a factor 2.
    confidence_threshold : confidence threshold for chunk registration
    max_residual_shift : max shift in pixels for all iterations except the first.
    margin_ratios : ratio that determines the margin size relative to the chunk size.
    min_chunk : minimal chunk size
    min_margin : minimum margin size
    save_memory : Moves data out of device memory as soon as possible during iterations
        to avoid memory overflows for very large images, incurs a largely negligible performance cost.
    force_numpy: Forces output model to be allocated with numpy arrays.
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

    with asection(f"Starting multi-scale registration with num_iterations={num_iterations}"):
        aprint(f"Confidence threshold: {confidence_threshold}, max residual shif: {max_residual_shift}")

        for i in range(num_iterations):

            # pre-apply transform from previous iterations:
            if vector_field is not None:
                with asection(f"Applying transform from previous iteration (save_memory={save_memory})"):
                    model = WarpRegistrationModel(vector_field, force_numpy=force_numpy)
                    _, image = model.apply_pair(image_a, image_b)
                    if save_memory:
                        image = Backend.to_numpy(image)
            else:
                image = image_b

            with asection(f"register_warp_nd {i}"):
                nb_div = 2**i
                # Note: the trick below is a 'ceil division' ceil(u/v) == -(-u//v)
                chunks = tuple(max(min_chunk, -(-s // nb_div)) for s in image_a.shape)
                margins = tuple(max(min_margin, int(c * r)) for c, r in zip(chunks, margin_ratios))
                aprint(f"register iteration: {i}, div= {nb_div}, chunks={chunks}, margins={margins}")
                model = register_warp_nd(
                    image_a, image, chunks=chunks, margins=margins, force_numpy=force_numpy, **kwargs
                )
                aprint(f"median confidence: {model.median_confidence()}")
                aprint(
                    f"median shift magnitude: {model.median_shift_magnitude(confidence_threshold=confidence_threshold)}"
                )
                eff_max_shift = max_residual_shift if (i > 0 and max_residual_shift is not None) else None
                model.clean(mode="median", confidence_threshold=confidence_threshold, max_shift=eff_max_shift)

                model_vector_field = Backend.to_backend(model.vector_field)
                model_confidence = Backend.to_backend(model.confidence)

            if vector_field is None:
                with asection("Initialise vector field and confidence arrays."):
                    splits = tuple(s // max(min_chunk, -(-s // (2 ** (num_iterations - 1)))) for s in image_a.shape)
                    aprint(f"final resolution = {splits}")
                    vector_field = xp.zeros(shape=splits + (ndim,), dtype=model_vector_field.dtype)
                    confidence = xp.zeros(shape=splits, dtype=model_confidence.dtype)

            if model.mean_confidence() > confidence_threshold // 2:
                with asection("Updating vector field and confidence array"):
                    scale_factors = tuple(s / ms for ms, s in zip(model_confidence.shape, confidence.shape))
                    aprint(f"Scaling obtained vector field and confidence arrays by a factor {scale_factors}")

                    scaled_vector_field = sp.ndimage.zoom(model_vector_field, zoom=scale_factors + (1,), order=1)
                    vector_field += scaled_vector_field

                    scaled_confidence = sp.ndimage.zoom(model_confidence, zoom=scale_factors, order=1)
                    confidence = xp.maximum(confidence, scaled_confidence)
            else:
                aprint("Scale ignored!")
                break

    model = WarpRegistrationModel(vector_field=vector_field, confidence=confidence, force_numpy=force_numpy)

    return model
