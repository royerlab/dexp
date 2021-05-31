import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy

from dexp.utils import xpArray


class Backend(ABC):
    """
    Some description.
    """

    def __init__(self):
        """ Instanciates an Image Processing backend

        """

    _local = threading.local()

    @staticmethod
    def reset():
        if hasattr(Backend._local, 'backend_stack'):
            backends = Backend._local.backend_stack.copy()
            backends.reverse()
            for backend in backends:
                backend.__exit__()
            Backend._local.backend_stack = []

    @staticmethod
    def current(raise_error_if_none: bool = False):

        if hasattr(Backend._local, 'backend_stack'):
            backend_stack = Backend._local.backend_stack
            if backend_stack is None or len(backend_stack) == 0:
                if raise_error_if_none:
                    raise RuntimeError("No backend available in current thread context")
                else:
                    # aprint("Warning: no backend available in current thread context! falling back to a numpy backend! ")
                    from dexp.processing.backends.numpy_backend import NumpyBackend
                    return NumpyBackend()
            backend = backend_stack[-1]
            return backend
        else:
            if raise_error_if_none:
                raise RuntimeError("No backend available in current thread context")
            else:
                # aprint("Warning: no backend available in current thread context! falling back to a numpy backend! ")
                from dexp.processing.backends.numpy_backend import NumpyBackend
                return NumpyBackend()

    @staticmethod
    def set(backend: 'Backend'):
        if not hasattr(Backend._local, 'backend_stack'):
            Backend._local.backend_stack = []
        Backend._local.backend_stack.append(backend)

    @staticmethod
    def to_numpy(array: xpArray, dtype=None, force_copy: bool = False) -> numpy.ndarray:
        return Backend.current()._to_numpy(array, dtype=dtype, force_copy=force_copy)

    @staticmethod
    def to_backend(array: xpArray, dtype=None, force_copy: bool = False) -> Any:
        return Backend.current()._to_backend(array, dtype=dtype, force_copy=force_copy)

    @staticmethod
    def get_xp_module(array: xpArray=None) -> Any:
        return Backend.current()._get_xp_module(array)

    @staticmethod
    def get_sp_module(array: xpArray=None) -> Any:
        return Backend.current()._get_sp_module(array)

    def __enter__(self):
        if not hasattr(Backend._local, 'backend_stack'):
            Backend._local.backend_stack = []
        Backend._local.backend_stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        Backend._local.backend_stack.pop()

    @abstractmethod
    def copy(self, *args, **kwargs):
        raise NotImplementedError('Method not implemented!')

    @abstractmethod
    def clear_memory_pool(self):
        import gc
        gc.collect()

    def close(self):
        """ Releases all ressources allocated/cached by backend (if can be done safely)

        """
        self.clear_memory_pool()

    @abstractmethod
    def _to_numpy(self, array: xpArray, dtype=None, force_copy: bool = False) -> numpy.ndarray:
        """ Converts backend array to numpy. If array is already a numpy array it is returned unchanged.

        Parameters
        ----------
        array : backend array to be converted
        dtype : coerce array to given dtype
        force_copy : forces the return array to be a copy

        Returns
        -------
        array converted to backend

        """
        raise NotImplementedError('Method not implemented!')

    @abstractmethod
    def _to_backend(self, array: xpArray, dtype=None, force_copy: bool = False) -> Any:
        """ Converts numpy array to backend array, if already backend array, then it is returned unchanged

        Parameters
        ----------
        array : numpy array to be converted
        dtype : coerce array to given dtype
        force_copy : forces the return array to be a copy
        """
        raise NotImplementedError('Method not implemented!')

    @abstractmethod
    def _get_xp_module(self, array: xpArray=None) -> Any:
        """ Returns the numpy-like module for a given array

        Parameters
        ----------
        array : array from which to get the numpy-like module


        Returns
        -------
        numpy-like module
        """
        raise NotImplementedError('Method not implemented!')

    @abstractmethod
    def _get_sp_module(self, array: xpArray=None) -> Any:
        """ Returns the scipy-like module for a given array

        Parameters
        ----------
        array : array from which to get the scipy-like module

        Returns
        -------
        scipy-like module
        """
        raise NotImplementedError('Method not implemented!')
