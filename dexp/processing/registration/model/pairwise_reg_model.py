from abc import ABC, abstractmethod

import numpy


class PairwiseRegistrationModel(ABC):

    def __init__(self):
        """ Instanciates a translation-only registration model

        """

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def to_json(self) -> str:
        pass

    @abstractmethod
    def apply(self, image_a, image_b, pad: bool = False) -> numpy.ndarray:
        """ Applies this pairwise registration model to the two given images.
        Two images will be returned as there might be need to pad both images.


        Parameters
        ----------
        backend : Backend to use for computation
        image_a, image_b : backend array to be converted
        pad : Set to True to add padding, False otherwise

        Returns
        -------
        image_a_reg, image_b_reg registered images

        """
        raise NotImplementedError("Not implemented")
