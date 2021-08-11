from typing import Any

import numpy
import scipy
from dask.array import Array

from dexp.processing.backends.backend import Backend
from dexp.utils import xpArray


class NumpyBackend(Backend):
    """
    NumpyBackend
    """

    def __init__(self, *args, **kwargs):
        """ Instanciates a Numpy-based Image Processing backend

        """
        ## Important: Leave this, this is to make sure that the ndimage package works properly!
        exec("import scipy.ndimage")

    def copy(self, *args, **kwargs):
        return NumpyBackend()

    def __str__(self):
        return "NumpyBackend"

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, type, value, traceback):
        super().__exit__(type, value, traceback)

    def clear_memory_pool(self):
        pass

    def _to_numpy(self, array: xpArray, dtype=None, force_copy: bool = False) -> numpy.ndarray:

        if isinstance(array, Array):
            return self._to_numpy(array.compute(), dtype=dtype, force_copy=force_copy)

        if dtype:
            return array.astype(dtype, copy=force_copy)
        elif force_copy:
            return numpy.asarray(array.copy())
        else:
            return numpy.asarray(array)

    def _to_backend(self, array: xpArray, dtype=None, force_copy: bool = False) -> Any:

        if isinstance(array, Array):
            return self._to_backend(array.compute(), dtype=dtype, force_copy=force_copy)

        if dtype:
            return array.astype(dtype, copy=force_copy)
        elif force_copy:
            return array.copy()
        else:
            return array

    def _get_xp_module(self, array: xpArray=None) -> Any:
        if array is None:
            return numpy
        else:
            try:
                import cupy
                return cupy.get_array_module(array)
            except ModuleNotFoundError:
                return numpy

    def _get_sp_module(self, array: xpArray=None) -> Any:
        if array is None:
            return scipy
        else:
            try:
                import cupyx
                return cupyx.scipy.get_array_module(array)
            except ModuleNotFoundError:
                return scipy
