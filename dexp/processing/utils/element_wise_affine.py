import numexpr

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend


def element_wise_affine(array, alpha, beta, sum_first=False, out=None):
    """
    Applies the affine function: alpha*x + beta to every value x of a given array. If sum_first is True, then alpha*(x + beta) is computed instead.

    Parameters
    ----------
    array : array to apply function to
    alpha : 'scale'
    beta : 'offset'
    sum_first : If True then alpha*(x + beta) is computed instead.
    out : if specified, element_wise_affine _might_ use the specified array as output for in-place operation.
          this is not garanteed, so you should always use the pattern: result = element_wise_affine(backend, a, array, alpha, beta, out=result).


    Returns
    -------
    Array: alpha*array + beta (or alpha*array + beta if sum_first is True)

    """

    array = Backend.to_backend(array)

    if type(Backend.current()) is NumpyBackend:
        if sum_first:
            return numexpr.evaluate("alpha*(array+beta)", casting='same_kind', out=out)
        else:
            return numexpr.evaluate("alpha*array+beta", casting='same_kind', out=out)

    elif type(Backend.current()) is CupyBackend:
        import cupy
        if sum_first:
            @cupy.fuse()
            def affine_function(_array, _alpha, _beta):
                return _alpha * (_array + _beta)

            return affine_function(array, alpha, beta)
        else:
            @cupy.fuse()
            def affine_function(_array, _alpha, _beta):
                return _alpha * _array + _beta

            return affine_function(array, alpha, beta)
