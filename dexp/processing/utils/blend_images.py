import numexpr

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend


def blend_images(array_a, array_b,
                 blend_map,
                 dtype=None):
    """
    Blends two arrays on the basis of a blend map. A value of 1 in the blend map signifies that the value of the first array is chosen,
    a value of 0 means that the value from teh second array is chosen instead. For values within ]0,1[ linear blending is performed.

    Parameters
    ----------
    array_a : first array to blend
    array_b : second array to blend
    blend_map : blend map with values within [0, 1]
    dtype   : dtype for returned array

    Returns
    -------
    Blended array.

    """

    if array_a.dtype != array_b.dtype:
        raise ValueError("Two arrays to blend must have same dtype!")

    if array_a.shape != array_b.shape:
        raise ValueError("Two arrays to blend must have same shape!")

    if dtype is None:
        dtype = array_a.dtype

    array_a = Backend.to_backend(array_a, dtype=dtype)
    array_b = Backend.to_backend(array_b, dtype=dtype)
    blend_map = Backend.to_backend(blend_map)

    backend = Backend.current()

    if type(backend) is NumpyBackend:
        a = array_a
        b = array_b
        m = blend_map
        return numexpr.evaluate("a*m+(1-m)*b")

    elif type(backend) is CupyBackend:
        import cupy
        @cupy.fuse()
        def blend_function(a, b, m):
            return a * m + (1 - m) * b

        return blend_function(array_a, array_b, blend_map)
