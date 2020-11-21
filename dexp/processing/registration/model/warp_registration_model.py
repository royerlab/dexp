import json
from typing import Any, Tuple

from dexp.processing.backends.backend import Backend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.model.pairwise_reg_model import PairwiseRegistrationModel


class WarpRegistrationModel(PairwiseRegistrationModel):

    def __init__(self,
                 vector_field,
                 confidence):
        """ Instantiates a translation registration model

        """
        super().__init__()
        self.vector_field = vector_field
        self.confidence = confidence

    def __str__(self):
        return f"WarpRegistrationModel(vector_field_shape={self.vector_field.shape}, confidence_shape={self.confidence.shape})"

    def to_json(self) -> str:
        return json.dumps({'type': 'warp', 'vector_field': self.vector_field, 'confidence_shape': self.confidence})

    def apply(self, backend: Backend, image_a, image_b, pad: bool = False) -> Tuple[Any, Any]:
        image_b_warped = warp(image=image_b,
                              vector_field=self.vector_field)

        return image_a, image_b_warped
