import numexpr

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend


def element_wise_affine(backend: Backend, array, alpha, beta):
    """
    Applies the affine function: alpha*x + beta to every value x of a given array.

    Parameters
    ----------
    backend : backend to use
    array : array to apply function to
    alpha : 'scale'
    beta : 'offset'

    Returns
    -------
    Array: alpha*array + beta

    """
    array = backend.to_backend(array)

    if type(backend) is NumpyBackend:
        a = array
        u = alpha
        v = beta
        return numexpr.evaluate("u*a+v")

    elif type(backend) is CupyBackend:
        import cupy
        @cupy.fuse()
        def affine_function(a,u, v):
            return u*a+v

        return affine_function(array, alpha, beta)
