from dexp.processing.utils.fit_shape import fit_to_shape
from dexp.utils.backends import Backend
from dexp.utils.testing.testing import execute_both_backends


@execute_both_backends
def test_fit_shape():
    xp = Backend.get_xp_module()

    array_1 = xp.random.uniform(0, 1, size=(31, 10, 17))
    array_2 = xp.random.uniform(0, 1, size=(32, 9, 18))

    array_2_fit = fit_to_shape(array_2.copy(), shape=array_1.shape)

    assert array_2_fit is not array_2
    assert array_2_fit.shape == array_1.shape
