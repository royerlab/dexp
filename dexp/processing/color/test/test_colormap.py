from dexp.processing.color.demo.demo_colormap import demo_colormap
from dexp.utils.backends import CupyBackend, NumpyBackend


def test_colormap_numpy():
    with NumpyBackend():
        _test_colormap()


def test_colormap_cupy():
    try:
        with CupyBackend():
            _test_colormap()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_colormap():
    demo_colormap(length_xy=64, display=False)
