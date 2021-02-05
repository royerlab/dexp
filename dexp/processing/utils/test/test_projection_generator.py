from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.projection_generator import projection_generator


def test_projection_generator_numpy():
    with NumpyBackend():
        _test_projection_generator()


def test_projection_generator_cupy():
    try:
        with CupyBackend():
            _test_projection_generator()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_projection_generator():
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    image = xp.random.rand(12, 14, 17, 19)

    projections = list(projection_generator(image))
    assert len(projections) == 6

    for u, v, projection in projection_generator(image):
        assert 0 <= u <= 3
        assert 0 <= v <= 3
        assert u < v

        assert projection.shape[0] == image.shape[u]
        assert projection.shape[1] == image.shape[v]
