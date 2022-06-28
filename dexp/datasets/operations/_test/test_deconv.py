from arbol import aprint

from dexp.datasets.operations.demo.demo_deconv import _demo_deconv
from dexp.utils.backends import CupyBackend


def test_deconv_cupy():
    try:
        with CupyBackend():
            _demo_deconv(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
