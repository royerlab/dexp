from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def test_backend_context():
    Backend.reset()
    assert type(Backend.current(raise_error_if_none=False)) == NumpyBackend

    with NumpyBackend() as backend_1:
        backend_c = Backend.current()
        assert backend_c == backend_1

        with NumpyBackend() as backend_2:
            backend_c = Backend.current()
            assert backend_c == backend_2
            assert backend_1 != backend_2

        backend_c = Backend.current()
        assert backend_c == backend_1
