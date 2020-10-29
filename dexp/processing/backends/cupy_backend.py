from abc import abstractmethod
from typing import Any

import numpy

from dexp.processing.backends.backend import Backend


class CupyBackend(Backend):

    def __init__(self):
        """ Instanciates a Numpy-based Image Processing backend

        """

    def close(self):
        #Nothing to do
        pass

    def to_numpy(self, array, dtype=None) -> numpy.ndarray:
        import cupy
        if cupy.get_array_module(array) == cupy:
            array = cupy.asnumpy(array)
            if dtype:
                array = array.astype(dtype, copy=False)
            return array
        else:
            return array

    def to_backend(self, array, dtype=None) -> Any:
        import cupy
        if dtype:
            array = array.astype(dtype, copy=False)

        if cupy.get_array_module(array) == cupy:
            return array
        else:
            return cupy.asarray(array)

    def get_xp_module(self, array) -> Any:
        import cupy
        return cupy.get_array_module(array)

    def get_sp_module(self, array) -> Any:
        import cupyx
        return cupyx.scipy.get_array_module(array)

















