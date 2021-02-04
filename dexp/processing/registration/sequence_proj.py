from typing import Tuple

from dask.array import Array

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.sequence_registration_model import SequenceRegistrationModel


def sequence_stabilisation_proj(image: 'Array',
                                axis: int = 0,
                                **kwargs
                                ) -> SequenceRegistrationModel:
    """
    Computes a sequence stabilisation model for an image sequence indexed along a specified axis.
    Stabilises all 2D projections and then combines the information to figure out what is the best
    stabilisation for the whole nD image


    Parameters
    ----------
    image: image to stabilise
    axis: sequence axis along which to stabilise image
    kwargs: argument passthrough to the 'sequence_stabilisation' method.

    Returns
    -------
    sequence registration model

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # TODO: compute all projections and call method below...


def sequence_stabilisation_proj_(projections: dict[Tuple[int, int], 'Array'],
                                 axis: int = 0,
                                 **kwargs
                                 ) -> SequenceRegistrationModel:
    pass
    # TODO: uses the provided projections (u,v) -> projection to compute the model
