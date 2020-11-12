import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend


def nan_to_zero(backend: Backend, array, copy=True):
    """
    Replaces every nan in an array to zero. It might, or not, be able to operate in-place.
    To be safe, the returned array should always be used...

    Parameters
    ----------
    backend : Backend for computation
    array : array to replace NaNs with zeros.
    copy : True/False to suggest whether copy or in-place behaviour should occur.

    Returns
    -------
    Array for which NaNs have been replace by zero.

    """

    if type(backend) is NumpyBackend:
        xp = backend.get_xp_module()
        return xp.nan_to_num(array, copy=copy)
    elif type(backend) is CupyBackend:
        import cupy
        return cupy.nan_to_num(array)

