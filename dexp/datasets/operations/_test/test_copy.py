from arbol import aprint

from dexp.datasets.operations.demo.demo_copy import _demo_copy
from dexp.utils.backends import CupyBackend, NumpyBackend


def test_copy_numpy():
    with NumpyBackend():
        _demo_copy(display=False)


def test_copy_cupy():
    try:
        with CupyBackend():
            _demo_copy(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
