from typing import Any, Tuple, List

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.pairwise_reg_model import PairwiseRegistrationModel


class TranslationRegistrationModel(PairwiseRegistrationModel):

    def __init__(self, shift_vector: List[float], error: float = 0, integral: bool = False):
        """ Instanciates a Numpy-based Image Processing backend

        """
        super().__init__()
        self.shift_vector = shift_vector
        self.error = error
        self.integral = integral

    def __str__(self):
        return f"TranslationRegistrationModel(shift={self.shift_vector}, error={self.error}, integral={self.integral})"

    def get_shift_and_error(self):
        return self.shift_vector, self.error

    def apply(self, backend: Backend, image_a, image_b, pad: bool = False) -> Tuple[Any, Any]:

        xp = backend.get_xp_module()

        if self.integral:
            integral_shift_vector = tuple(int(round(shift)) for shift in self.shift_vector)

            padding_a = tuple(((0, abs(s)) if s >= 0 else (abs(s), 0)) for s in integral_shift_vector)
            padding_b = tuple(((0, abs(s)) if s < 0 else (abs(s), 0)) for s in integral_shift_vector)

            if pad:
                image_a = xp.pad(image_a, pad_width=padding_a)
                image_b = xp.pad(image_b, pad_width=padding_b)

            else:
                image_b = numpy.roll(image_b,
                                     shift=integral_shift_vector,
                                     axis=range(len(integral_shift_vector)))

            return image_a, image_b
        else:
            raise NotImplementedError("Not implemented!")
