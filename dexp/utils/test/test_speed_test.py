from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.utils.speed_test import perform_speed_test


def test_backend_context():
    perform_speed_test()
