from typing import Any, Tuple

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.pairwise_reg_model import PairwiseRegistrationModel


class Translation(PairwiseRegistrationModel):

    def __init__(self, shift_vector, integral: bool = True):
        """ Instanciates a Numpy-based Image Processing backend

        """
        super().__init__()
        self.shift_vector = shift_vector
        self.integral = integral

    def apply(self, backend: Backend, image_a, image_b) -> Tuple[Any, Any]:

        xp = backend.get_xp_module()

        if self.integral:
            integral_shift_vector = tuple(int(round(shift)) for shift in self.shift_vector)

            padding_a = (((0, abs(s)) if s >= 0 else (abs(s), 0)) for s in integral_shift_vector)
            padding_b = (((0, abs(s)) if s < 0 else (abs(s), 0)) for s in integral_shift_vector)

            image_a_padded = xp.pad(image_a, padding=padding_a)
            image_b_padded = xp.pad(image_b, padding=padding_b)


            image_b_padded = numpy.roll(image_b_padded,
                                        shift=integral_shift_vector,
                                        axis=range(len(integral_shift_vector)))

            return image_a_padded, image_b_padded
        else:
            raise NotImplementedError("Not implemented!")


















