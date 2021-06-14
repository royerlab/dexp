from arbol import aprint

from dexp.datasets.operations.demo.demo_deconv import _demo_deconv
from dexp.processing.backends.cupy_backend import CupyBackend


# deconv CLI only available for cuda
# def test_deconv_numpy():
#     with NumpyBackend():
#         _demo_deconv(display=False)


def test_deconv_cupy():
    try:
        with CupyBackend():
            _demo_deconv(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


