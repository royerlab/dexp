from dexp.utils import xpArray
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def nan_to_zero(array: xpArray, copy: bool = True) -> xpArray:
    """
    Replaces every nan in an array to zero. It might, or not, be able to operate in-place.
    To be safe, the returned array should always be used...

    Parameters
    ----------
    array : array to replace NaNs with zeros.
    copy : True/False to suggest whether copy or in-place behaviour should occur.

    Returns
    -------
    Array for which NaNs have been replace by zero.

    """
    # TODO: should we remove this function?
    backend = Backend.current()

    if type(backend) is NumpyBackend:
        xp = backend.get_xp_module()
        return xp.nan_to_num(array, copy=copy)
    elif type(backend) is CupyBackend:
        import cupy

        return cupy.nan_to_num(array)
