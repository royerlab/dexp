import inspect
from functools import wraps
from typing import Callable

import pytest

from dexp.utils.backends import CupyBackend, NumpyBackend, dispatch_data_to_backend
from dexp.utils.backends.cupy_backend import is_cupy_available


def _add_cuda_signature(func: Callable) -> Callable:
    """Adds `cuda` argument to the given function."""
    # inspect parameters
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # add new parameter
    params.insert(0, inspect.Parameter("cuda", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD))

    # overwrite old signature
    func.__signature__ = sig.replace(parameters=params)

    # add cuda parameter to function
    parametrizer = pytest.mark.parametrize("cuda", [False, True], ids=["cpu", "gpu"])
    func = parametrizer(func)
    return func


def execute_both_backends(func: Callable) -> Callable:
    """Helper function to execute both backends in a single function call."""

    @wraps(func)
    def wrapper(cuda: bool, *args, **kwargs):
        args = list(args)
        if cuda:
            try:
                with CupyBackend() as backend:
                    dispatch_data_to_backend(backend, args, kwargs)
                    func(*args, **kwargs)
            except ModuleNotFoundError:
                pytest.skip(f"Cupy not found. Skipping {func.__name__} gpu test.")
        else:
            with NumpyBackend() as backend:
                # it assumes that is a numpy array by default
                func(*args, **kwargs)

    return _add_cuda_signature(wrapper)


def cupy_only(func: Callable) -> Callable:
    """Helper function to skip test function is cupy is not found."""

    @pytest.mark.skipif(not is_cupy_available(), reason=f"Cupy not found. Skipping {func.__name__} gpu test.")
    @wraps(func)
    def _func(*args, **kwargs):
        return func(*args, **kwargs)

    return _func
