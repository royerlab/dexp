from itertools import combinations
from typing import List

import numpy as np
from numpy.typing import ArrayLike

from dexp.utils import xpArray
from dexp.utils.backends import Backend

__all__ = [
    "first_derivative_func",
    "first_derivative_kernels",
    "second_derivative_func",
    "second_derivative_kernels",
    "derivative_axes",
]


def line_derivative_kernels(dim: int, template: ArrayLike) -> List[xpArray]:
    kernels = []
    for axis in range(dim):
        shape = np.ones(dim, dtype=int)
        shape[axis] = 3
        D = np.zeros(shape, dtype=np.float32)
        slicing = tuple(slice(None) if i == axis else 0 for i in range(dim))
        D[slicing] = template
        kernels.append(D)
    return kernels


def diagonal_derivative_kernels(dim: int, template: ArrayLike) -> List[xpArray]:
    kernels = []
    for axes in combinations(range(dim), 2):
        shape = np.ones(dim, dtype=int)
        shape[list(axes)] = 3
        D = np.zeros(shape, dtype=np.float32)
        slicing = tuple(slice(None) if i in axes else 0 for i in range(dim))
        D[slicing] = template
        kernels.append(D)

    return kernels


def first_derivative_kernels(dim: int) -> List[xpArray]:
    kernels = []
    line = np.array([1, -1, 0])
    kernels += line_derivative_kernels(dim, line)
    diagonal = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    kernels += diagonal_derivative_kernels(dim, diagonal)
    return kernels


def second_derivative_kernels(dim: int) -> List[xpArray]:
    kernels = []
    line = np.array([1, -2, 1])
    kernels += line_derivative_kernels(dim, line)
    diagonal = np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]])
    kernels += diagonal_derivative_kernels(dim, diagonal)
    return kernels


def derivative_axes(dim: int) -> List[int]:
    # Must be kept in the same order as the other derivative functions
    return list(range(dim)) + list(combinations(range(dim), 2))


def first_derivative_func(array: xpArray, axes: int, transpose: bool) -> xpArray:
    xp = Backend.get_xp_module()

    left_slice = [slice(None) for _ in range(array.ndim)]
    right_slice = [slice(None) for _ in range(array.ndim)]

    if isinstance(axes, int):
        axes = (axes,)

    for axis in axes:
        left_slice[axis] = slice(None, -1)
        right_slice[axis] = slice(1, None)

    if transpose:
        left_slice, right_slice = right_slice, left_slice

    left_slice = tuple(left_slice)
    right_slice = tuple(right_slice)

    out = xp.zeros_like(array)
    out[right_slice] = array[left_slice] - array[right_slice]

    return out


def second_derivative_func(array: xpArray, axes: int, transpose: bool) -> xpArray:
    return -1 * first_derivative_func(first_derivative_func(array, axes, transpose), axes, not transpose)
