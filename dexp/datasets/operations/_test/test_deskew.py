from arbol import aprint

from dexp.datasets.operations.demo.demo_deskew import _demo_deskew
from dexp.utils.backends import CupyBackend


def test_deskew_cupy():
    try:
        with CupyBackend():
            _demo_deskew(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
