from typing import Sequence, Tuple

import numpy

from dexp.processing.registration.model.sequence_registration_model import (
    SequenceRegistrationModel,
)
from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.processing.registration.sequence import image_stabilisation
from dexp.processing.utils.projection_generator import projection_generator
from dexp.utils import xpArray
from dexp.utils.backends import Backend


def image_stabilisation_proj(
    image: xpArray, axis: int = 0, projection_type: str = "max-min", **kwargs
) -> SequenceRegistrationModel:
    """
    Computes a sequence stabilisation model for an image sequence indexed along a specified axis.
    Instead of running a full nD registration, this uses projections instead, economising memory and compute.
    Stabilises all 2D projections and then combines the information to figure out what is the best
    stabilisation for the whole nD image


    Parameters
    ----------
    image: image to stabilise
    axis: sequence axis along which to stabilise image
    projection_type : Projection type to use when in 'projection' mode: 'mean', 'min', 'max', 'max-min'
    kwargs: argument passthrough to the 'sequence_stabilisation' method.

    Returns
    -------
    sequence registration model

    """
    xp = Backend.get_xp_module()

    ndim = image.ndim
    image = xp.moveaxis(image, axis, 0)
    projections = projection_generator(image, axis_range=(1, ndim), projection_type=projection_type)

    return image_stabilisation_proj_(projections=projections, ndim=ndim - 1, **kwargs)


def image_stabilisation_proj_(
    projections: Sequence[Tuple], ndim: int, keep_best: bool = True, debug_output: str = None, **kwargs
) -> SequenceRegistrationModel:
    """
    Same as 'sequence_stabilisation_proj' but takes projections instead,
    usefull if the projections are already available.

    Parameters
    ----------
    projections: List of projections as a list of tuples: (u, v, projection) where u and v are the axis indices
        and projection is an array where the first axis is the stabilisation axis.
    ndim: number of dimensions of original array
    keep_best: keep only the best n stabilised projections
    kwargs: argument passthrough to the 'sequence_stabilisation' method.

    Returns
    -------
    Sequence registration model

    """
    # we stabilise along each projection:
    seq_reg_models = list(
        (
            u - 1,
            v - 1,
            image_stabilisation(
                image=projection,
                axis=0,
                debug_output=f"{u - 1}_{v - 1}_{debug_output}" if debug_output is not None else debug_output,
                **kwargs,
            ),
        )
        for u, v, projection in projections
    )

    # Let's sort the models by decreasing confidence:
    seq_reg_models = sorted(seq_reg_models, key=lambda t: t[2].overall_confidence(), reverse=True)

    if keep_best:
        # Only keep the best models, but enough to cover all coordinates:
        counts = numpy.zeros(shape=(ndim,), dtype=numpy.float32)
        selected_seq_reg_models = []
        for u, v, seq_reg_model in seq_reg_models:
            counts[u] += 1
            counts[v] += 1
            selected_seq_reg_models.append((u, v, seq_reg_model))
            if numpy.all(counts > 0):
                break
        seq_reg_models = selected_seq_reg_models

    # first we figure out the length of the sequence, and figure out the kind of model:
    _, _, model = seq_reg_models[0]
    length = len(model.model_list)

    if type(model.model_list[0]) == TranslationRegistrationModel:

        # Create a new model list:
        fused_model_list = [
            TranslationRegistrationModel(shift_vector=numpy.zeros(shape=(ndim,), dtype=numpy.float32))
            for _ in range(length)
        ]

        # Iterate through the sequence updating the fused model:
        for tp in range(length):

            # current fused model:
            fused_model = fused_model_list[tp]

            # keeps track of how many shift values we add here:
            counts = numpy.zeros(shape=(ndim,), dtype=numpy.float32)

            # We extract the models for the current time point (tp):
            models = list(tuple((u, v, seq_reg_model.model_list[tp])) for u, v, seq_reg_model in seq_reg_models)

            for u, v, model in models:
                model: TranslationRegistrationModel

                # we add the shifts at the right places:
                if not keep_best or counts[u] == 0:
                    fused_model.shift_vector[u] += model.shift_vector[0]
                    counts[u] += 1
                if not keep_best or counts[v] == 0:
                    fused_model.shift_vector[v] += model.shift_vector[1]
                    counts[v] += 1

            # ... so we can compute averages:
            fused_model.shift_vector /= counts

        model: SequenceRegistrationModel = SequenceRegistrationModel(model_list=fused_model_list)

        if debug_output is not None:
            model.plot(debug_output + "_fused_model")

        return model

    else:
        raise ValueError("Pairwise registration model not supported!")
