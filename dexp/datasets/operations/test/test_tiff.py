from arbol import aprint

from dexp.datasets.operations.demo.demo_tiff import _demo_tiff
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend


def test_copy_numpy():
    with NumpyBackend():
        _demo_tiff(display=False)

def test_copy_cupy():
    try:
        with CupyBackend():
            _demo_tiff(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


