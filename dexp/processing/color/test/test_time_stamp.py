from dexp.processing.color.demo.demo_time_stamp import demo_time_stamp
from dexp.utils.backends import CupyBackend, NumpyBackend


def test_time_stamp_numpy():
    with NumpyBackend():
        _test_time_stamp()


def test_time_stamp_cupy():
    try:
        with CupyBackend():
            _test_time_stamp()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_time_stamp():
    demo_time_stamp(display=False)
