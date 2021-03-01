from arbol import aprint, asection

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend


def BestBackend(*args, **kwargs):
    with asection("Determining best backend..."):
        try:
            aprint("Checking availability of Cupy and testing.")
            import cupy
            with cupy.cuda.Device(0):
                array = cupy.array([1, 2, 3])
                assert cupy.median(array) == 2
            aprint("Cupy backend available and functional!")
            return CupyBackend(*args, **kwargs)

        except Exception:
            aprint("Cupy module not found or not functional! ignored!")
            return NumpyBackend(*args, **kwargs)
