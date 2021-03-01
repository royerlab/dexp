from arbol import aprint

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.video.demo.demo_overlay import demo_overlay


def test_overlay_numpy():
    with NumpyBackend():
        demo_overlay(display=False)


def test_overlay_cupy():
    try:
        with CupyBackend():
            demo_overlay(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
