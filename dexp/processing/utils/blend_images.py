import numexpr

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend


def blend_images(backend: Backend, array_a, array_b, blend_map):
    """
    Blends two arrays on the basis of a blend map. A value of 1 in the blend map signifies that the value of the first array is chosen,
    a value of 0 means that the value from teh second array is chosen instead. For values within ]0,1[ linear blending is performed.

    Parameters
    ----------
    backend : backend to use
    array_a : first array to blend
    array_b : second array to blend
    blend_map : blend map with values within [0, 1]

    Returns
    -------
    Blended array.

    """
    array_a = backend.to_backend(array_a)
    array_b = backend.to_backend(array_b)
    blend_map = backend.to_backend(blend_map)

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
