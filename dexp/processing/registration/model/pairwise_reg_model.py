from abc import ABC, abstractmethod

import numpy

from dexp.processing.backends.backend import Backend


class PairwiseRegistrationModel(ABC):

    def __init__(self):
        """ Instanciates a translation-only registration model

        """


    @abstractmethod
    def apply(self, backend: Backend, image_a, image_b) -> numpy.ndarray:
        """ Applies this pairwise registration model to the two given images.
        Two images will be returned as there might be need to pad both images.


        Parameters
        ----------
        backend : Backend to use for computation
        image_a, image_b : backend array to be converted

        Returns
        -------
        image_a_reg, image_b_reg registered images

        """
        raise NotImplementedError("Not implemented")















