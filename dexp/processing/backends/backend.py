from abc import ABC, abstractmethod
from typing import Sequence, Any

import numpy


class Backend(ABC):

    def __init__(self):
        """ Instanciates an Image Processing backend

        """

    @abstractmethod
    def close(self):
        """ Releases all ressources allocated/cached by backend (if can be done safely)

        """
        pass

    @abstractmethod
    def to_numpy(self, array, dtype=None) -> numpy.ndarray:
        """ Converts backend array to numpy. If array is already a numpy array it is returned unchanged.

        Parameters
        ----------
        array : backend array to be converted
        dtype : coerce array to given dtype

        Returns
        -------
        array converted to backend

        """
        pass

    @abstractmethod
    def to_backend(self, array, dtype=None) -> Any:
        """ Converts numpy array to backend array, if already backend array, then it is returned unchanged

        Parameters
        ----------
        array : numpy array to be converted
        dtype : coerce array to given dtype
        """
        pass


    @abstractmethod
    def get_xp_module(self, array) -> Any:
        """ Returns the numpy-like module for a given array

        Parameters
        ----------
        array : array from which to get the numpy-like module

        Returns
        -------
        numpy-like module
        """
        pass

    @abstractmethod
    def get_sp_module(self, array) -> Any:
        """ Returns the scipy-like module for a given array

        Parameters
        ----------
        array : array from which to get the scipy-like module

        Returns
        -------
        scipy-like module
        """
        pass















