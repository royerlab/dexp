import numpy as np

from dexp.processing.utils.nan_to_zero import nan_to_zero
from dexp.utils.backends import Backend
from dexp.utils.testing.testing import execute_both_backends


@execute_both_backends
def test_nan_to_zero_shape():
    xp = Backend.get_xp_module()

    array_1 = xp.random.uniform(0, 1, size=(32, 10, 17)).astype(dtype=xp.float32)
    array_2 = xp.random.uniform(0, 1, size=(32, 10, 17)).astype(dtype=xp.float32)
    array_2[array_2 < 0.1] = 0

    with np.errstate(divide="ignore"):
        array_1 /= array_2

    assert xp.isinf(array_1).any()
    array_1 = nan_to_zero(array_1)
    assert not xp.isinf(array_1).any()
