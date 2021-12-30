from arbol import aprint

from dexp.processing.backends import CupyBackend, NumpyBackend
from dexp.processing.deskew.demo.demo_yang_deskew import _yang_deskew


def test_yang_deskew_numpy():
    with NumpyBackend():
        _yang_deskew(length=48, display=False)


def test_yang_deskew_cupy():
    try:
        with CupyBackend():
            _yang_deskew(length=48, display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
