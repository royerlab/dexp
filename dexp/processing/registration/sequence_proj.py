from typing import Tuple, Sequence

import numpy
from dask.array import Array

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.sequence_registration_model import SequenceRegistrationModel
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.sequence import image_stabilisation
from dexp.processing.utils.projection_generator import projection_generator


def image_stabilisation_proj(image: 'Array',
                             axis: int = 0,
                             projection_type: str = 'max-min',
                             **kwargs
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
    sp = Backend.get_sp_module()

    ndim = image.ndim
    image = xp.moveaxis(image, axis, 0)
    projections = projection_generator(image,
                                       axis_range=(1, ndim),
                                       projection_type=projection_type)

    return image_stabilisation_proj_(projections=projections,
                                     ndim=ndim - 1)


def image_stabilisation_proj_(projections: Sequence[Tuple],
                              ndim: int,
                              keep_best: bool = True,
                              **kwargs
                              ) -> SequenceRegistrationModel:
    """
    Same as 'sequence_stabilisation_proj' but takes projections instead, usefull if the projections are already available.

    Parameters
    ----------
    projections: List of projections as a list of tuples: (u, v, projection) where u and v are the axis indices and projection is an array where the first axis is the stabilisation axis.
    ndim: number of dimensions of original array
    keep_best: keep only the best n stabilised projections
    kwargs: argument passthrough to the 'sequence_stabilisation' method.

    Returns
    -------
    Sequence registration model

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # we stabilise along each projection:
    seq_reg_models = list((u - 1, v - 1, image_stabilisation(image=projection, axis=0, **kwargs)) for u, v, projection in projections)

    # first we figure out the length of the sequence, and figure out the kind of model:
    _, _, model = seq_reg_models[0]
    length = len(model.model_list)

    if type(model.model_list[0]) == TranslationRegistrationModel:

        # second, we create a new model list:
        fused_model_list = [TranslationRegistrationModel(shift_vector=numpy.zeros(shape=(ndim,), dtype=numpy.float32)) for _ in range(length)]

        # third, we iterate through the sequence updating the fused model:
        for tp in range(length):

            # current fused model:
            fused_model = fused_model_list[tp]

            # keep track of how many shift values we add here:
            counts = numpy.zeros(shape=(ndim,), dtype=numpy.float32)

            # We extract the models for the current time point (tp):
            models = list(tuple((u, v, seq_reg_model.model_list[tp])) for u, v, seq_reg_model in seq_reg_models)

            # Let's sort the models by decreasing confidence:
            models = sorted(models, key=lambda t: t[2].overall_confidence(), reverse=True)

            for u, v, model in models:
                model: TranslationRegistrationModel

                # ... add the shifts at the right places:
                fused_model.shift_vector[u] += model.shift_vector[0]
                fused_model.shift_vector[v] += model.shift_vector[1]

                # and keep track of the number of shifts added ...
                counts[u] += 1
                counts[v] += 1

                if keep_best and Backend.get_xp_module(counts).all(counts > 0):
                    # we stop if all coefficients of the shift vector are covered:
                    break

            # ... so we can compute averages:
            fused_model.shift_vector /= counts

        return SequenceRegistrationModel(model_list=fused_model_list)

    else:
        raise ValueError("Pairwise registration model not supported!")
