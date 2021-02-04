import json
from typing import Tuple, Union, Sequence

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.pairwise_registration_model import PairwiseRegistrationModel


class TranslationRegistrationModel(PairwiseRegistrationModel):

    def __init__(self,
                 shift_vector: Union[Sequence[float], numpy.ndarray],
                 confidence: Union[numpy.ndarray, float] = 1,
                 integral: bool = False,
                 force_numpy: bool = True):

        """ Instantiates a translation registration model

        Parameters
        ----------
        shift_vector : Relative shift between two images
        confidence : registration confidence: a float within [0, 1] which conveys how confident is the registration.
        A value of 0 means no confidence, a value of 1 means perfectly confident.
        force_numpy : when creating this object, you have the option of forcing the use of numpy array instead of the current backend arrays.
        integral : True if shifts are snapped to integer values, False otherwise

        """
        super().__init__()
        xp = Backend.get_xp_module()

        if force_numpy:
            self.shift_vector = Backend.to_numpy(0 if shift_vector is None else shift_vector)
            self.confidence = Backend.to_numpy(0 if confidence is None else confidence)
        else:
            self.shift_vector = xp.asarray(0 if shift_vector is None else shift_vector)
            self.confidence = xp.asarray(0 if confidence is None else confidence)

        self.integral = integral

    def __str__(self):
        return f"TranslationRegistrationModel(shift={self.shift_vector}, confidence={self.confidence}, integral={self.integral})"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.shift_vector == other.shift_vector).all() and (self.confidence == other.confidence).all() and self.integral == other.integral
        else:
            return False

    def to_json(self) -> str:
        return json.dumps({'type': 'translation', 'translation': self.shift_vector.tolist(), 'integral': self.integral, 'confidence': self.confidence.tolist()})

    def to_numpy(self) -> 'TranslationRegistrationModel':
        self.shift_vector = Backend.to_numpy(self.shift_vector)
        self.confidence = Backend.to_numpy(self.confidence)
        return self

    def padding(self):
        integral_shift_vector = tuple(int(round(float(shift))) for shift in self.shift_vector)
        padding = tuple(((0, abs(s)) if s < 0 else (abs(s), 0)) for s in integral_shift_vector)
        return padding

    def overall_confidence(self) -> float:
        return float(self.confidence)

    def change_relative_to(self, other) -> float:
        return float(numpy.linalg.norm(Backend.to_numpy(other.shift_vector) - Backend.to_numpy(self.shift_vector)))

    def get_shift_and_confidence(self):
        return self.shift_vector, self.confidence

    def apply(self, image, pad: bool = False) -> 'Array':

        image = Backend.to_backend(image)

        integral_shift_vector = tuple(int(round(float(shift))) for shift in self.shift_vector)

        if pad:
            xp = Backend.get_xp_module()
            padding = tuple(((0, abs(s)) if s < 0 else (abs(s), 0)) for s in integral_shift_vector)
            image = xp.pad(image, pad_width=padding)
            return image

        else:
            if self.integral:
                image = numpy.roll(image,
                                   shift=integral_shift_vector,
                                   axis=range(len(integral_shift_vector)))

                return image
            else:
                sp = Backend.get_sp_module()
                image = sp.ndimage.shift(image,
                                         shift=self.shift_vector,
                                         order=1)
                return image

    def apply_pair(self, image_a, image_b, pad: bool = False) -> Tuple['Array', 'Array']:

        xp = Backend.get_xp_module()

        integral_shift_vector = tuple(int(round(float(shift))) for shift in self.shift_vector)

        if pad:
            padding_a = tuple(((0, abs(s)) if s >= 0 else (abs(s), 0)) for s in integral_shift_vector)
            padding_b = tuple(((0, abs(s)) if s < 0 else (abs(s), 0)) for s in integral_shift_vector)

            image_a = Backend.to_backend(image_a)
            image_b = Backend.to_backend(image_b)

            image_a = xp.pad(image_a, pad_width=padding_a)
            image_b = xp.pad(image_b, pad_width=padding_b)

            return image_a, image_b

        else:
            return image_a, self.apply(image_b, pad=pad)
