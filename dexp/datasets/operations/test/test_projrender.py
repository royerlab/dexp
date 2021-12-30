from arbol import aprint

from dexp.datasets.operations.demo.demo_projrender import _demo_projrender
from dexp.processing.backends import CupyBackend, NumpyBackend


def test_projrender_numpy():
    with NumpyBackend():
        _demo_projrender(display=False)


def test_stabilize_cupy():
    try:
        with CupyBackend():
            _demo_projrender(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
