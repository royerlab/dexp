from arbol import aprint

from dexp.utils.backends.cupy_backend import CupyBackend
from dexp.utils.backends.numpy_backend import NumpyBackend


def BestBackend(*args, **kwargs):
    try:
        import cupy

        deviced_id = kwargs.get("device_id", 0)
        with cupy.cuda.Device(deviced_id):
            array = cupy.array([1, 2, 3])
            assert cupy.median(array) == 2
        return CupyBackend(*args, **kwargs)

    except Exception:
        aprint("Cupy module not found or not functional! ignored!")
        return NumpyBackend(*args, **kwargs)
