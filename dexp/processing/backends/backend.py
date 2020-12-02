import threading
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any

import numpy
import psutil


class Backend(ABC):

    def __init__(self):
        """ Instanciates an Image Processing backend

        """

    _local = threading.local()
    _pool = ThreadPoolExecutor(max_workers=psutil.cpu_count())

    @staticmethod
    def reset():
        if hasattr(Backend._local, 'backend_stack'):
            backends = Backend._local.backend_stack.copy()
            backends.reverse()
            for backend in backends:
                backend.__exit__()
            Backend._local.backend_stack = []

    @staticmethod
    def current(raise_error_if_none: bool = True):

        if hasattr(Backend._local, 'backend_stack'):
            backend_stack = Backend._local.backend_stack
            if backend_stack is None or len(backend_stack) == 0:
                if raise_error_if_none:
                    raise RuntimeError("No backend available in current thread context")
                else:
                    return None
            backend = backend_stack[-1]
            return backend
        else:
            if raise_error_if_none:
                raise RuntimeError("No backend available in current thread context")
            else:
                return None

    @staticmethod
    def to_numpy(array, dtype=None, force_copy: bool = False) -> numpy.ndarray:
        return Backend.current()._to_numpy(array, dtype=dtype, force_copy=force_copy)

    @staticmethod
    def to_backend(array, dtype=None, force_copy: bool = False) -> Any:
        return Backend.current()._to_backend(array, dtype=dtype, force_copy=force_copy)

    @staticmethod
    def get_xp_module(array=None) -> Any:
        return Backend.current()._get_xp_module(array)

    @staticmethod
    def get_sp_module(array=None) -> Any:
        return Backend.current()._get_sp_module(array)

    @staticmethod
    def submit(self, *args, **kwargs):
        self._pool.submit(*args, **kwargs)

    def __enter__(self):
        if not hasattr(Backend._local, 'backend_stack'):
            Backend._local.backend_stack = []
        Backend._local.backend_stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        Backend._local.backend_stack.pop()

    def synchronise(self):
        """ Synchronises backend computation to this call, i.e. call to this method will block until all computation on backend (and its corresponding device) are finished.

        """
        pass

    @abstractmethod
    def close(self):
        """ Releases all ressources allocated/cached by backend (if can be done safely)

        """
        pass

    @abstractmethod
    def _to_numpy(self, array, dtype=None, force_copy: bool = False) -> numpy.ndarray:
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
        pass

    @abstractmethod
    def _to_backend(self, array, dtype=None, force_copy: bool = False) -> Any:
        """ Converts numpy array to backend array, if already backend array, then it is returned unchanged

        Parameters
        ----------
        array : numpy array to be converted
        dtype : coerce array to given dtype
        force_copy : forces the return array to be a copy
        """
        pass

    @abstractmethod
    def _get_xp_module(self, array=None) -> Any:
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
    def _get_sp_module(self, array=None) -> Any:
        """ Returns the scipy-like module for a given array

        Parameters
        ----------
        array : array from which to get the scipy-like module

        Returns
        -------
        scipy-like module
        """
        pass
