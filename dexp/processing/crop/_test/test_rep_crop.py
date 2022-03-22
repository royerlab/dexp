# flake8: noqa
from dexp.processing.crop.demo.demo_rep_crop import _demo_representative_crop
from dexp.utils.backends import CupyBackend, NumpyBackend


def test_representative_crop_numpy():
    with NumpyBackend():
        _test_representative_crop()


def test_representative_crop_cupy():
    try:
        with CupyBackend():
            _test_representative_crop()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_representative_crop():
    _demo_representative_crop(display=False, fast_mode=True)
    _demo_representative_crop(display=False, fast_mode=False)
