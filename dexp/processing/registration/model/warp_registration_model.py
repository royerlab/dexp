import json
from typing import Tuple

import numpy

from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.model.pairwise_registration_model import (
    PairwiseRegistrationModel,
)
from dexp.utils import xpArray
from dexp.utils.backends import Backend


class WarpRegistrationModel(PairwiseRegistrationModel):
    def __init__(self, vector_field, confidence=None, force_numpy: bool = False):
        """Warp registration model.
        A warp registration model consists in a vector field of shape (w,1), (h,w,2),
            or (d,h,w,3) and a confidence matrix of shape (w), (h,w), or (d,h,w),
        The vector field represents a warp transform, and the confidence matrix the confidence score
            -- within [0, 1] -- of each vector in the vector field.

        Parameters
        ----------
        vector_field : vector field for the warp transform
        confidence : confidence matrix
        force_numpy : when creating this object, you have the option of forcing
            the use of numpy array instead of the current backend arrays.
        This is usefull to avoid memory fragmentation (e.g. on GPU).
        """
        super().__init__()

        xp = Backend.get_xp_module()

        if force_numpy:
            self.vector_field = Backend.to_numpy(0 if vector_field is None else vector_field)
            self.confidence = Backend.to_numpy(0 if confidence is None else confidence)
        else:
            self.vector_field = xp.asarray(0 if vector_field is None else vector_field)
            self.confidence = xp.asarray(0 if confidence is None else confidence)

    def __str__(self):
        return (
            f"WarpRegistrationModel(vector_field_shape={self.vector_field.shape}, "
            + f"confidence_shape={self.confidence.shape})"
        )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.vector_field == other.vector_field).all() and (self.confidence == other.confidence).all()
        else:
            return False

    def to_json(self) -> str:
        return json.dumps(
            {"type": "warp", "vector_field": self.vector_field.tolist(), "confidence": self.confidence.tolist()}
        )

    def to_numpy(self) -> "WarpRegistrationModel":
        self.vector_field = Backend.to_numpy(self.vector_field)
        self.confidence = Backend.to_numpy(self.confidence)
        return self

    def padding(self):
        return tuple((0, 0) for _ in range(self.confidence.ndim))

    def overall_confidence(self) -> float:
        return float(self.median_confidence())

    def change_relative_to(self, other) -> float:
        ndim = self.vector_field.ndim
        robust_average = numpy.median(
            numpy.linalg.norm(Backend.to_numpy(other.vector_field) - Backend.to_numpy(self.vector_field), axis=ndim - 1)
        )
        return float(robust_average)

    def clean(self, max_shift: float = None, confidence_threshold: float = 0.1, mode: str = "mean"):
        """
        Cleans the vector field by in-filling vectors of low confidence with the median of
        neighbooring higher-confidence vectors.

        Parameters
        ----------
        confidence_threshold : confidence threshold below which a vector is deamed unreliable.
        mode : How to propagate high-confidence values, can be 'mean' or 'median'
        """
        xp = Backend.get_xp_module()
        sp = Backend.get_sp_module()

        # First we take care of excessive shifts above the 'max_shift' limit:
        vector_field = Backend.to_backend(self.vector_field, force_copy=True)
        original_vector_field = Backend.to_backend(self.vector_field, force_copy=True)
        confidence = Backend.to_backend(self.confidence, force_copy=True)

        if max_shift is not None:
            model_vector_field_norm = xp.linalg.norm(vector_field, axis=-1)
            too_much = model_vector_field_norm > max_shift
            confidence[too_much] = 0.00013579  # close to zero but recognisable

        # Second we fill out low confidence entries:
        num_iterations = 2 * numpy.sum(confidence.shape)  # enough iterations to reach a steady-state
        mask = confidence > confidence_threshold
        vector_field[~mask] = 0
        for i in range(num_iterations):
            if mode == "median":
                vector_field = sp.ndimage.median_filter(
                    vector_field, size=(3,) * (confidence.ndim) + (1,), mode="nearest"
                )
            elif mode == "mean":
                vector_field = sp.ndimage.uniform_filter(
                    vector_field, size=(3,) * (confidence.ndim) + (1,), mode="nearest"
                )
            else:
                raise ValueError("Unsupported mode")
            # we make sure to keep the high-confidence vectors unchanged:
            vector_field[mask] = original_vector_field[mask]

        self.vector_field = vector_field
        self.confidence = confidence

    def apply(
        self,
        image,
        pad: bool = False,
        vector_field_upsampling: int = 2,
        vector_field_upsampling_order: int = 1,
        mode: str = "border",
        internal_dtype=None,
    ) -> xpArray:

        image_warped = warp(
            image=image,
            vector_field=self.vector_field,
            vector_field_upsampling=vector_field_upsampling,
            vector_field_upsampling_order=vector_field_upsampling_order,
            mode=mode,
            internal_dtype=internal_dtype,
        )

        return image_warped

    def apply_pair(self, image_a, image_b, **kwargs) -> Tuple[xpArray, xpArray]:

        return image_a, self.apply(image_b, **kwargs)

    def median_confidence(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        xp = Backend.get_xp_module(self.confidence)
        return xp.median(self.confidence)

    def mean_confidence(self):
        """

        Parameters
        ----------
        backend

        Returns
        -------

        """
        xp = Backend.get_xp_module(self.confidence)
        return xp.mean(self.confidence)

    def median_shift_magnitude(self, confidence_threshold: float = 0.5):
        """

        Parameters
        ----------
        backend
        confidence_threshold

        Returns
        -------

        """
        xp = Backend.get_xp_module(self.confidence)
        ndim = self.vector_field.ndim
        vector_field = self.vector_field.astype(dtype=numpy.float32)
        norms = xp.linalg.norm(vector_field, axis=ndim - 1)
        confidence = self.confidence
        norms = norms[confidence > confidence_threshold]
        if norms.size == 0:
            return 0
        elif norms.size == 1:
            return float(norms)
        else:
            return xp.median(norms)
