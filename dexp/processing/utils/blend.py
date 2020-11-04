import numexpr

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend


def blend_arrays(backend: Backend, array_a, array_b, blend_map):
    array_a = backend.to_backend(array_a)
    array_b = backend.to_backend(array_b)
    blend_map = backend.to_backend(blend_map)

    if type(backend) is NumpyBackend:
        a = array_a
        b = array_b
        m = blend_map
        numexpr.evaluate("a*m+(1-m)*b")
    elif type(backend) is CupyBackend:
        import cupy
        @cupy.fuse()
        def blend_function(a, b, m):
            return a * m + (1 - m) * b

        return blend_function(array_a, array_b, blend_map)
