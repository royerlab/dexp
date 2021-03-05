from arbol import aprint

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend


def BestBackend(*args, **kwargs):
    try:
        import cupy
        with cupy.cuda.Device(0):
            array = cupy.array([1, 2, 3])
            assert cupy.median(array) == 2
        return CupyBackend(*args, **kwargs)

    except Exception:
        aprint("Cupy module not found or not functional! ignored!")
        return NumpyBackend(*args, **kwargs)
