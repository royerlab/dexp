import inspect
from functools import wraps
from typing import Any, Callable, Dict, Iterable

import pytest

from dexp.utils import xpArray
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def _maybe_to_backend(backend: Backend, obj: Any) -> Any:
    if isinstance(obj, xpArray):
        return backend.to_backend(obj)
    else:
        return obj


def dispatch_data_to_backend(backend: Backend, args: Iterable, kwargs: Dict) -> None:
    """Function to move arrays to backend."""
    for i, v in enumerate(args):
        args[i] = _maybe_to_backend(backend, v)

    for k, v in kwargs.items():
        kwargs[k] = _maybe_to_backend(backend, v)


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
                    dispatch_data_to_backend(backend, kwargs)
                    func(*args, **kwargs)
            except ModuleNotFoundError:
                pytest.skip(f"Cupy not found. Skipping {func.__name__} gpu test.")
        else:
            with NumpyBackend() as backend:
                dispatch_data_to_backend(backend, args, kwargs)
                func(*args, **kwargs)

    return _add_cuda_signature(wrapper)
