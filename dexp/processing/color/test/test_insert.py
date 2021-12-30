from dexp.processing.backends import CupyBackend, NumpyBackend
from dexp.processing.color.demo.demo_insert import demo_insert


def test_insert_numpy():
    with NumpyBackend():
        _test_insert()


def test_insert_cupy():
    try:
        with CupyBackend():
            _test_insert()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_insert():
    demo_insert(display=False)
