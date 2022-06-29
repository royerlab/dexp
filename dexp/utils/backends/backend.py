import importlib
import threading
import types
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from dexp.utils import xpArray


class Backend(ABC):
    """
    Some description.

    Methods
    -------
    close:
        Releases resources allocated by backend.
    """

    def __init__(self):
        """Instanciates an Image Processing backend"""

    _local = threading.local()

    @staticmethod
    def reset():
        if hasattr(Backend._local, "backend_stack"):
            backends = Backend._local.backend_stack.copy()
            backends.reverse()
            for backend in backends:
                backend.__exit__()
            Backend._local.backend_stack = []

    @staticmethod
    def current(raise_error_if_none: bool = False) -> "Backend":

        if hasattr(Backend._local, "backend_stack"):
            backend_stack = Backend._local.backend_stack
            if backend_stack is None or len(backend_stack) == 0:
                if raise_error_if_none:
                    raise RuntimeError("No backend available in current thread context")
                else:
                    from dexp.utils.backends.numpy_backend import NumpyBackend

                    return NumpyBackend()
            backend = backend_stack[-1]
            return backend
        else:
            if raise_error_if_none:
                raise RuntimeError("No backend available in current thread context")
            else:
                # aprint("Warning: no backend available in current thread context! falling back to a numpy backend! ")
                from dexp.utils.backends import NumpyBackend

                return NumpyBackend()

    @staticmethod
    def set(backend: "Backend"):
        if not hasattr(Backend._local, "backend_stack"):
            Backend._local.backend_stack = []
        Backend._local.backend_stack.append(backend)

    @staticmethod
    def to_numpy(array: xpArray, dtype=None, force_copy: bool = False) -> np.ndarray:
        return Backend.current()._to_numpy(array, dtype=dtype, force_copy=force_copy)

    @staticmethod
    def to_backend(array: xpArray, dtype=None, force_copy: bool = False) -> Any:
        return Backend.current()._to_backend(array, dtype=dtype, force_copy=force_copy)

    @staticmethod
    def get_xp_module(array: Optional[xpArray] = None) -> types.ModuleType:
        return Backend.current()._get_xp_module(array)

    @staticmethod
    def get_sp_module(array: Optional[xpArray] = None) -> types.ModuleType:
        return Backend.current()._get_sp_module(array)

    @staticmethod
    def get_skimage_submodule(submodule: str, array: Optional[xpArray] = None) -> types.ModuleType:
        return Backend.current()._get_skimage_submodule(submodule, array)

    def __enter__(self):
        if not hasattr(Backend._local, "backend_stack"):
            Backend._local.backend_stack = []
        Backend._local.backend_stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        Backend._local.backend_stack.pop()

    @abstractmethod
    def copy(self, *args, **kwargs):
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def clear_memory_pool(self):
        import gc

        gc.collect()

    def close(self):
        """Releases all ressources allocated/cached by backend (if can be done safely)"""
        self.clear_memory_pool()

    @abstractmethod
    def _to_numpy(self, array: xpArray, dtype=None, force_copy: bool = False) -> np.ndarray:
        """Converts backend array to numpy. If array is already a numpy array it is returned unchanged.

        Parameters
        ----------
        array : backend array to be converted
        dtype : coerce array to given dtype
        force_copy : forces the return array to be a copy

        Returns
        -------
        array converted to backend

        """
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    def _to_backend(self, array: xpArray, dtype=None, force_copy: bool = False) -> Any:
        """Converts numpy array to backend array, if already backend array, then it is returned unchanged

        Parameters
        ----------
        array : numpy array to be converted
        dtype : coerce array to given dtype
        force_copy : forces the return array to be a copy
        """
        raise NotImplementedError("Method not implemented!")

    def _get_xp_module(self, array: xpArray) -> types.ModuleType:
        """Returns the numpy-like module for a given array

        Parameters
        ----------
        array : array from which to get the numpy-like module


        Returns
        -------
        numpy-like module
        """
        try:
            import cupy

            return cupy.get_array_module(array)
        except ModuleNotFoundError:
            return np

    def _get_sp_module(self, array: xpArray) -> types.ModuleType:
        """Returns the scipy-like module for a given array

        Parameters
        ----------
        array : array from which to get the scipy-like module

        Returns
        -------
        scipy-like module
        """
        try:
            import cupyx

            return cupyx.scipy.get_array_module(array)
        except ModuleNotFoundError:
            import scipy

            return scipy

    def _get_skimage_submodule(self, submodule: str, array: xpArray) -> types.ModuleType:
        """Returns the skimage-like module for a given array (skimage or cucim)

        Parameters
        ----------
        array : xpArray, optional
            Reference array
        Returns
        -------
        types.ModuleType
            skimage-like module
        """

        # avoiding try-catch
        try:
            import cupy

            if isinstance(array, cupy.ndarray):
                return importlib.import_module(f"cucim.skimage.{submodule}")
        except ModuleNotFoundError:
            return importlib.import_module(f"skimage.{submodule}")

        return importlib.import_module(f"skimage.{submodule}")

    def synchronise(self) -> None:
        pass


def _maybe_to_backend(backend: Backend, obj: Any) -> Any:

    if isinstance(obj, np.ndarray):
        # this must be first because arrays are iterables
        return backend.to_backend(obj)

    if isinstance(obj, Dict):
        dispatch_data_to_backend(backend, [], obj)
        return obj

    elif isinstance(obj, (List, Tuple)):
        obj = list(obj)
        dispatch_data_to_backend(backend, obj, {})
        return obj

    else:
        return obj


def dispatch_data_to_backend(backend: Backend, args: Iterable, kwargs: Dict) -> None:
    """Function to move arrays to backend INPLACE!"""
    for i, v in enumerate(args):
        args[i] = _maybe_to_backend(backend, v)

    for k, v in kwargs.items():
        kwargs[k] = _maybe_to_backend(backend, v)
