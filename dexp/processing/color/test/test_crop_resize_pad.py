from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.color.demo.demo_border import demo_border
from dexp.processing.color.demo.demo_crop_resize_pad import demo_crop_resize_pad


def test_crop_resize_pad_numpy():
    with NumpyBackend():
        _test_crop_resize_pad()


def test_crop_resize_pad_cupy():
    try:
        with CupyBackend():
            _test_crop_resize_pad()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_crop_resize_pad():
    demo_crop_resize_pad(display=False)
