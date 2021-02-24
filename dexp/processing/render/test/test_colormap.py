from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.render.demo.demo_colormap import demo_colormap


def test_colormap_numpy():
    with NumpyBackend():
        test_colormap()


def test_colormap_cupy():
    try:
        with CupyBackend():
            test_colormap()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def test_colormap():
    demo_colormap(length_xy=64, display=False)
