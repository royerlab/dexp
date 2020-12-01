from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.nan_to_zero import nan_to_zero


def test_nan_to_zero_numpy():
    with NumpyBackend():
        _test_nan_to_zero_shape()


def test_nan_to_zero_cupy():
    try:
        with CupyBackend():
            _test_nan_to_zero_shape()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_nan_to_zero_shape(length_xy=128):
    xp = Backend.get_xp_module()

    array_1 = xp.random.uniform(0, 1, size=(32, 10, 17)).astype(dtype=xp.float32)
    array_2 = xp.random.uniform(0, 1, size=(32, 10, 17)).astype(dtype=xp.float32)
    array_2[array_2 < 0.1] = 0

    array_1 /= array_2

    assert xp.isinf(array_1).any()  # xp.isnan(array_1).any() or
    array_1 = nan_to_zero(array_1)
    assert not xp.isinf(array_1).any()  # not xp.isnan(array_1).any() and
