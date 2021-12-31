from arbol import aprint

from dexp.utils.backends import CupyBackend, NumpyBackend
from dexp.video.demo.demo_blend import demo_blend


def test_blend_numpy():
    with NumpyBackend():
        demo_blend(display=False)


def test_blend_cupy():
    try:
        with CupyBackend():
            demo_blend(display=False)

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
