from arbol import aprint

from dexp.datasets.operations.demo.demo_deskew import _demo_deskew
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend


def test_deskew_numpy():
    with NumpyBackend():
        _demo_deskew(display=False)

def test_deskew_cupy():
    try:
        with CupyBackend():
            _demo_deskew(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


