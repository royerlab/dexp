from abc import ABC, abstractmethod
from typing import Tuple

from dexp.utils import xpArray


class PairwiseRegistrationModel(ABC):
    def __init__(self):
        """Instanciates a translation-only registration model"""

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def padding(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def overall_confidence(self) -> float:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def to_json(self) -> str:
        pass

    @abstractmethod
    def to_numpy(self):
        pass

    @abstractmethod
    def change_relative_to(self, other) -> float:
        pass

    @abstractmethod
    def apply(self, image, pad: bool = False) -> xpArray:
        """Applies this pairwise registration model to a single image.


        Parameters
        ----------
        image : two images to register to each other.
        pad : Set to True to add padding, False otherwise

        Returns
        -------
        image for which the registration transform has been applied.

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def apply_pair(self, image_a, image_b, pad: bool = False) -> Tuple[xpArray, xpArray]:
        """Applies this pairwise registration model to the two given images.
        Two images will be returned as there might be need to pad both images.


        Parameters
        ----------
        image_a, image_b : two images to register to each other.
        pad : Set to True to add padding, False otherwise

        Returns
        -------
        image_a_reg, image_b_reg registered images. In some cases, and if possible,
            only the second image will be modified.

        """
        raise NotImplementedError("Not implemented")
