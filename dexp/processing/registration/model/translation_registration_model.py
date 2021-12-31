import json
from typing import Sequence, Tuple, Union

import numpy

from dexp.processing.registration.model.pairwise_registration_model import (
    PairwiseRegistrationModel,
)
from dexp.utils import xpArray
from dexp.utils.backends import Backend


class TranslationRegistrationModel(PairwiseRegistrationModel):
    def __init__(
        self,
        shift_vector: Union[Sequence[float], numpy.ndarray],
        confidence: Union[numpy.ndarray, float] = 1.0,
        force_numpy: bool = True,
    ):

        """Instantiates a translation registration model

        Parameters
        ----------
        shift_vector : Relative shift between two images
        confidence : registration confidence: a float within [0, 1] which conveys how confident is the registration.
        A value of 0 means no confidence, a value of 1 means perfectly confident.
        force_numpy : when creating this object, you have the option of forcing the use of numpy array
            instead of the current backend arrays.

        """
        super().__init__()
        xp = Backend.get_xp_module()

        if force_numpy:
            self.shift_vector = Backend.to_numpy(0 if shift_vector is None else shift_vector)
            self.confidence = Backend.to_numpy(0 if confidence is None else confidence)
        else:
            self.shift_vector = xp.asarray(0 if shift_vector is None else shift_vector)
            self.confidence = xp.asarray(0 if confidence is None else confidence)

    def __str__(self):
        return f"TranslationRegistrationModel(shift={self.shift_vector}, confidence={self.confidence})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.shift_vector == other.shift_vector).all() and (self.confidence == other.confidence).all()
        else:
            return False

    def __sub__(self, other: "TranslationRegistrationModel") -> "TranslationRegistrationModel":
        if isinstance(other, self.__class__):
            return TranslationRegistrationModel(
                shift_vector=self.shift_vector - other.shift_vector, confidence=self.confidence
            )
        else:
            raise ValueError("Expected another `TranslationRegistrationModel` at `__sub__`.")

    def __add__(self, other: "TranslationRegistrationModel") -> "TranslationRegistrationModel":
        if isinstance(other, self.__class__):
            return TranslationRegistrationModel(
                shift_vector=self.shift_vector + other.shift_vector,
                confidence=(self.confidence + other.confidence) / 2.0,
            )
        else:
            raise ValueError("Expected another `TranslationRegistrationModel` at `__add__`.")

    def copy(self) -> "TranslationRegistrationModel":
        return TranslationRegistrationModel(self.shift_vector, self.confidence)

    def to_json(self) -> str:
        return json.dumps(
            {"type": "translation", "translation": self.shift_vector.tolist(), "confidence": self.confidence.tolist()}
        )

    def to_numpy(self) -> "TranslationRegistrationModel":
        self.shift_vector = Backend.to_numpy(self.shift_vector)
        self.confidence = Backend.to_numpy(self.confidence)
        return self

    def integral_shift(self):
        return tuple(int(round(float(shift))) for shift in self.shift_vector)

    def padding(self):
        integral_shift_vector = self.integral_shift()
        padding = tuple(((0, abs(s)) if s > 0 else (abs(s), 0)) for s in integral_shift_vector)
        return padding

    def overall_confidence(self) -> float:
        return float(self.confidence)

    def change_relative_to(self, other) -> float:
        return float(numpy.linalg.norm(Backend.to_numpy(other.shift_vector) - Backend.to_numpy(self.shift_vector)))

    def get_shift_and_confidence(self):
        return self.shift_vector, self.confidence

    def apply(self, image, integral: bool = True, pad: bool = False) -> xpArray:
        """
        Applies the translation model to the given image, possibly by padding the image.

        Parameters
        ----------
        image: image to apply translation to.
        integral: True to snap translation to pixels, False to perform subpixel interpolation.
        pad: True to translate by padding, False for wrapped translation (image shape remains unchanged)

        Returns
        -------
        translated image, with or without padding.

        """
        image = Backend.to_backend(image)

        integral_shift_vector = self.integral_shift()

        if pad:
            xp = Backend.get_xp_module()
            padding = tuple(((0, abs(s)) if s < 0 else (abs(s), 0)) for s in integral_shift_vector)
            image = xp.pad(image, pad_width=padding)
            return image

        else:
            if integral:
                image = numpy.roll(image, shift=integral_shift_vector, axis=range(len(integral_shift_vector)))

                return image
            else:
                sp = Backend.get_sp_module()
                image = sp.ndimage.shift(image, shift=self.shift_vector, order=1)
                return image

    def apply_pair(self, image_a, image_b, integral: bool = True, pad: bool = False) -> Tuple[xpArray, xpArray]:
        """
        Applies the translation model to an image pair, possibly by padding the image.

        Parameters
        ----------
        image_a: First image
        image_b: Second image
        integral: True to snap translation to pixels, False to perform subpixel interpolation.
        pad: True to translate by padding, False for wrapped translation (image shape remains unchanged).

        Note: padding implies integral translation.

        Returns
        -------
        Both images registered
        """
        xp = Backend.get_xp_module()

        if pad:
            integral_shift_vector = self.integral_shift()

            padding_a = tuple(((0, abs(s)) if s >= 0 else (abs(s), 0)) for s in integral_shift_vector)
            padding_b = tuple(((0, abs(s)) if s < 0 else (abs(s), 0)) for s in integral_shift_vector)

            image_a = Backend.to_backend(image_a)
            image_b = Backend.to_backend(image_b)

            # necessary, otherwise the padding blows up the memory
            Backend.current().clear_memory_pool()

            image_a = xp.pad(image_a, pad_width=padding_a)
            image_b = xp.pad(image_b, pad_width=padding_b)

            return image_a, image_b

        else:
            return image_a, self.apply(image_b, integral=integral, pad=False)
