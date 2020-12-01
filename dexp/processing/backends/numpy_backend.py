from typing import Any

import numpy
import scipy

from dexp.processing.backends.backend import Backend


class NumpyBackend(Backend):

    def __init__(self):
        """ Instanciates a Numpy-based Image Processing backend

        """
        ## Important: Leave this, this is to make sure that the ndimage package works properly!
        exec("import scipy.ndimage")

    def __str__(self):
        return "NumpyBackend"

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, type, value, traceback):
        super().__exit__(type, value, traceback)

    def close(self):
        # Nothing to do
        pass

    def to_numpy(self, array, dtype=None, force_copy: bool = False) -> numpy.ndarray:
        if dtype:
            return array.astype(dtype, copy=force_copy)
        elif force_copy:
            return array.copy()
        else:
            return array

    def to_backend(self, array, dtype=None, force_copy: bool = False) -> Any:
        if dtype:
            return array.astype(dtype, copy=force_copy)
        elif force_copy:
            return array.copy()
        else:
            return array

    def get_xp_module(self, array=None) -> Any:
        return numpy

    def get_sp_module(self, array=None) -> Any:
        return scipy
