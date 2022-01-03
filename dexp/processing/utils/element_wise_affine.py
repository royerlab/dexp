from typing import Optional

import numexpr

from dexp.utils import xpArray
from dexp.utils.backends import Backend, CupyBackend


def element_wise_affine(
    array: xpArray, alpha: float, beta: float, sum_first: bool = False, out: Optional[xpArray] = None
) -> xpArray:

    """
    Applies the affine function: alpha*x + beta to every value x of a given array.
    If sum_first is True, then alpha*(x + beta) is computed instead.

    Parameters
    ----------
    array : array to apply function to
    alpha : 'scale'
    beta : 'offset'
    sum_first : If True then alpha*(x + beta) is computed instead.
    out : if specified, element_wise_affine _might_ use the specified array as output for in-place operation.
        this is not garanteed, so you should always use the pattern:
        result = element_wise_affine(backend, a, array, alpha, beta, out=result).


    Returns
    -------
    Array: alpha*array + beta (or alpha*array + beta if sum_first is True)

    """

    array = Backend.to_backend(array)

    if isinstance(Backend.current(), CupyBackend):
        import cupy

        if sum_first:

            @cupy.fuse()
            def affine_function(_array, _alpha, _beta):
                return _alpha * (_array + _beta)

        else:

            @cupy.fuse()
            def affine_function(_array, _alpha, _beta):
                return _alpha * _array + _beta

        return affine_function(array, alpha, beta)

    else:
        if sum_first:
            return numexpr.evaluate("alpha*(array+beta)", casting="same_kind", out=out)
        else:
            return numexpr.evaluate("alpha*array+beta", casting="same_kind", out=out)
