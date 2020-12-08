import json
from typing import Any, Tuple, Union, Sequence

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.pairwise_reg_model import PairwiseRegistrationModel


class TranslationRegistrationModel(PairwiseRegistrationModel):

    def __init__(self,
                 shift_vector: Union[Sequence[float], numpy.ndarray],
                 confidence: float = 1,
                 integral: bool = False):

        """ Instantiates a translation registration model

        Parameters
        ----------
        shift_vector : Relative shift between two images
        confidence : registration confidence: a float within [0, 1] which conveys how confident is the registration.
        A value of 0 means no confidence, a value of 1 means perfectly confident.
        integral : True if shifts are snapped to integer values, False otherwise

        """
        super().__init__()
        self.shift_vector = Backend.to_numpy(shift_vector)
        self.confidence = Backend.to_numpy(confidence)
        self.integral = integral

    def __str__(self):
        return f"TranslationRegistrationModel(shift={self.shift_vector}, confidence={self.confidence}, integral={self.integral})"

    def to_json(self) -> str:
        return json.dumps({'type': 'translation', 'translation': self.shift_vector.tolist(), 'integral': self.integral, 'confidence': self.confidence.tolist()})

    def get_shift_and_confidence(self):
        return self.shift_vector, self.confidence

    def apply(self, image_a, image_b, pad: bool = False) -> Tuple[Any, Any]:

        xp = Backend.get_xp_module()
        sp = Backend.get_sp_module()

        integral_shift_vector = tuple(int(round(shift)) for shift in self.shift_vector)

        if pad:
            padding_a = tuple(((0, abs(s)) if s >= 0 else (abs(s), 0)) for s in integral_shift_vector)
            padding_b = tuple(((0, abs(s)) if s < 0 else (abs(s), 0)) for s in integral_shift_vector)

            image_a = xp.pad(image_a, pad_width=padding_a)
            image_b = xp.pad(image_b, pad_width=padding_b)

            return image_a, image_b

        else:
            if self.integral:
                image_b = numpy.roll(image_b,
                                     shift=integral_shift_vector,
                                     axis=range(len(integral_shift_vector)))

                return image_a, image_b
            else:
                image_b = sp.ndimage.shift(image_b,
                                           shift=self.shift_vector,
                                           order=1)
                return image_a, image_b
