from typing import Tuple, Sequence

import numpy
from dask.array import Array

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.sequence_registration_model import SequenceRegistrationModel
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.sequence import sequence_stabilisation
from dexp.processing.utils.projection_generator import projection_generator


def sequence_stabilisation_proj(image: 'Array',
                                axis: int = 0,
                                projection_type: str = 'mean',
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
    projection_type : Projection type to use when in 'projection' mode: 'mean', 'min', 'max'
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

    return sequence_stabilisation_proj_(projections=projections,
                                        ndim=ndim - 1)


def sequence_stabilisation_proj_(projections: Sequence[Tuple],
                                 ndim: int,
                                 **kwargs
                                 ) -> SequenceRegistrationModel:
    """
    Same as 'sequence_stabilisation_proj' but takes projections instead, usefull if the projections are already available.

    Parameters
    ----------
    projections:
    ndim: number of dimensions of original array
    kwargs: argument passthrough to the 'sequence_stabilisation' method.

    Returns
    -------
    Sequence registration model

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    seq_reg_models = list((u - 1, v - 1, sequence_stabilisation(image=projection, axis=0, **kwargs)) for u, v, projection in projections)

    # first we figure out the length of the sequence, and figure out the kind of model:
    _, _, model = seq_reg_models[0]
    length = len(model.model_list)

    if type(model.model_list[0]) == TranslationRegistrationModel:

        # second, we create a new model list:
        fused_model_list = [TranslationRegistrationModel(shift_vector=numpy.zeros(shape=(ndim,), dtype=numpy.float32)) for _ in range(length)]

        # third, we iterate through the sequence updating the fused model:
        for i in range(length):

            # current fused model:
            fused_model = fused_model_list[i]

            # keep track of how many shift values we add here:
            counts = numpy.zeros(shape=(ndim,), dtype=numpy.float32)

            for u, v, seq_reg_model in seq_reg_models:
                # for each seq reg model we...
                model: TranslationRegistrationModel = seq_reg_model.model_list[i]

                # ... add the shifts at the right places:
                fused_model.shift_vector[u] += model.shift_vector[0]
                fused_model.shift_vector[v] += model.shift_vector[1]

                # and keep track of the number of shifts added ...
                counts[u] += 1
                counts[v] += 1

            # ... so we can compute averages:
            fused_model.shift_vector /= counts

        return SequenceRegistrationModel(model_list=fused_model_list)

    else:
        raise ValueError("Pairwise registration model not supported!")
