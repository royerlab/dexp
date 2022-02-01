from typing import Callable

from dexp.utils import xpArray
from dexp.utils.backends import Backend


def apply_func(array: xpArray, *, func: Callable, axis: int, **kwargs) -> xpArray:

    xp = Backend.get_xp_module(array)

    results = []
    for i in range(array.shape[axis]):
        chunk = xp.take(array, i, axis=axis)
        results.append(func(chunk, **kwargs))

    return xp.stack(results, axis=axis)


def axis_wise(axis: int) -> xpArray:
    def decorator(func: Callable) -> xpArray:
        def wrapper(*args, **kwargs) -> xpArray:
            return apply_func(*args, **kwargs, func=func, axis=axis)

        return wrapper

    return decorator
