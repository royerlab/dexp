from abc import abstractmethod
from typing import Any

import numpy


from dexp.processing.backends.backend import Backend


class CupyBackend(Backend):

    def __init__(self, device=0, enable_cub: bool = True, enable_cutensor: bool = True):
        """ Instanciates a Numpy-based Image Processing backend

        """
        self.device = device
        from cupy.cuda import cub, cutensor
        cub.available = enable_cub
        cutensor.available = enable_cutensor

        ## Leave this:
        import cupyx.scipy.ndimage

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
            if dtype:
                array = array.astype(dtype, copy=False)
            return array

    def to_backend(self, array, dtype=None) -> Any:
        import cupy

        if cupy.get_array_module(array) == cupy:
            if dtype:
                array = array.astype(dtype, copy=False)
            return array
        else:
            with cupy.cuda.Device(self.device):
                return cupy.asarray(array, dtype=dtype)

    def get_xp_module(self, array=None) -> Any:
        if array is not None:
            import cupy
            return cupy.get_array_module(array)
        else:
            import cupy
            return cupy

    def get_sp_module(self, array=None) -> Any:
        if array is not None:
            import cupyx
            return cupyx.scipy.get_array_module(array)
        else:
            import cupyx
            return cupyx.scipy


















