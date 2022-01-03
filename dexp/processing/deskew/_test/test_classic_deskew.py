from arbol import aprint

from dexp.processing.deskew.demo.demo_classic_deskew import _classic_deskew
from dexp.utils.backends import CupyBackend, NumpyBackend


def test_classic_deskew_numpy():
    with NumpyBackend():
        _classic_deskew(length=48, display=False)


def test_classic_deskew_cupy():
    try:
        with CupyBackend():
            _classic_deskew(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
