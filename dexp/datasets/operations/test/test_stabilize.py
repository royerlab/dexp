from arbol import aprint

from dexp.datasets.operations.demo.demo_stabilize import _demo_stabilize
from dexp.processing.backends.cupy_backend import CupyBackend


# def test_stabilize_numpy():
#     with NumpyBackend():
#         _demo_stabilize(display=False)


def test_stabilize_cupy():
    try:
        with CupyBackend():
            _demo_stabilize(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
