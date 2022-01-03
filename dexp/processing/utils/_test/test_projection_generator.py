from dexp.processing.utils.projection_generator import projection_generator
from dexp.utils.backends import Backend
from dexp.utils.testing.testing import execute_both_backends


@execute_both_backends
def test_projection_generator():
    xp = Backend.get_xp_module()

    image = xp.random.rand(12, 14, 17, 19)

    projections = list(projection_generator(image))
    assert len(projections) == 6

    for u, v, projection in projection_generator(image):
        assert 0 <= u <= 3
        assert 0 <= v <= 3
        assert u < v

        assert projection.shape[0] == image.shape[u]
        assert projection.shape[1] == image.shape[v]
