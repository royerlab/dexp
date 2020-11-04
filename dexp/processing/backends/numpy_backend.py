from typing import Any

import numpy
import scipy

from dexp.processing.backends.backend import Backend


class NumpyBackend(Backend):

    def __init__(self):
        """ Instanciates a Numpy-based Image Processing backend

        """
        ## Leave this:

    def close(self):
        # Nothing to do
        pass

    def to_numpy(self, array, dtype=None, copy: bool = False) -> numpy.ndarray:
        if dtype:
            return array.astype(dtype, copy=copy)
        elif copy:
            return array.copy()
        else:
            return array

    def to_backend(self, array, dtype=None, copy: bool = False) -> Any:
        if dtype:
            return array.astype(dtype, copy=copy)
        elif copy:
            return array.copy()
        else:
            return array

    def get_xp_module(self, array=None) -> Any:
        return numpy

    def get_sp_module(self, array=None) -> Any:
        return scipy
