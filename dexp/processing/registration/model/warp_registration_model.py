import json
from typing import Any, Tuple

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.model.pairwise_reg_model import PairwiseRegistrationModel


class WarpRegistrationModel(PairwiseRegistrationModel):

    def __init__(self,
                 vector_field,
                 confidence=None):
        """ Instantiates a translation registration model

        """
        super().__init__()
        self.vector_field = vector_field
        self.confidence = confidence

    def __str__(self):
        return f"WarpRegistrationModel(vector_field_shape={self.vector_field.shape}, confidence_shape={self.confidence.shape})"

    def to_json(self) -> str:
        return json.dumps({'type': 'warp', 'vector_field': numpy.asarray(self.vector_field).tolist(), 'confidence': self.confidence})

    def median_confidence(self, backend: Backend):
        xp = backend.get_xp_module(self.confidence)
        return xp.median(self.confidence)

    def mean_confidence(self, backend: Backend):
        xp = backend.get_xp_module(self.confidence)
        return xp.mean(self.confidence)

    def median_shift_magnitude(self, backend: Backend, confidence_threshold: float = 0.7):
        xp = backend.get_xp_module(self.confidence)
        norms = xp.linalg.norm(self.vector_field, axis=-1)
        norms *= self.confidence > confidence_threshold
        return xp.median(norms)

    def clean(self,
              backend: Backend,
              confidence_threshold: float = 0.1,
              mode: str = 'mean'):
        """
        Cleans the vector field by in-filling vectors of low confidence with the median of neighbooring higher-confidence vectors.
        Parameters
        ----------
        backend : backend to use for computation
        confidence_threshold : confidence threshold below which a vector is deamed unreliable.
        mode : How to propagate high-confidence values, can be 'mean' or 'median'
        """
        sp = backend.get_sp_module()
        num_iterations = 1 + numpy.max(self.confidence.shape)
        mask = self.confidence > confidence_threshold
        vector_field = self.vector_field.copy()
        vector_field[~mask] = 0
        vector_field = backend.to_backend(vector_field)
        for i in range(num_iterations):
            if mode == 'median':
                vector_field = sp.ndimage.median_filter(vector_field, size=(3,) * (self.confidence.ndim) + (1,))
            elif mode == 'mean':
                vector_field = sp.ndimage.uniform_filter(vector_field, size=(3,) * (self.confidence.ndim) + (1,))
            else:
                raise ValueError("Unsupported mode")
            # we make sure to keep the high-confidence vectors unchanged:
            vector_field[mask] = self.vector_field[mask]

        self.vector_field = vector_field

    def apply(self, backend: Backend,
              image_a, image_b,
              vector_field_upsampling: int = 2,
              vector_field_upsampling_order: int = 1,
              mode: str = 'border',
              internal_dtype=None) -> Tuple[Any, Any]:

        image_b_warped = warp(backend,
                              image=image_b,
                              vector_field=self.vector_field,
                              vector_field_upsampling=vector_field_upsampling,
                              vector_field_upsampling_order=vector_field_upsampling_order,
                              mode=mode,
                              internal_dtype=internal_dtype)

        return image_a, image_b_warped
