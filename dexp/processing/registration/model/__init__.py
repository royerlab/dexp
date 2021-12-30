from typing import Union

from dexp.processing.registration.model.pairwise_registration_model import (
    PairwiseRegistrationModel,
)
from dexp.processing.registration.model.sequence_registration_model import (
    SequenceRegistrationModel,
)

RegistrationModel = Union[SequenceRegistrationModel, PairwiseRegistrationModel]
