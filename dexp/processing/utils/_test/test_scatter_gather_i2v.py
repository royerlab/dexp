import numpy as np
from arbol import aprint

from dexp.processing.utils.scatter_gather_i2v import scatter_gather_i2v
from dexp.utils.backends import Backend
from dexp.utils.testing.testing import execute_both_backends
from dexp.utils.timeit import timeit


@execute_both_backends
def test_scatter_gather_i2v(ndim=3, length_xy=128, splits=4):
    xp = Backend.get_xp_module()
    rng = np.random.default_rng()

    image1 = rng.uniform(0, 1, size=(length_xy,) * ndim)
    image2 = rng.uniform(0, 1, size=(length_xy,) * ndim)

    def f(x, y):
        return xp.stack([x.min(), x.max()]), xp.stack([y.max(), y.mean(), y.min()])

    with timeit("scatter_gather(f)"):
        chunks = (length_xy // splits,) * ndim
        result1, result2 = scatter_gather_i2v((image1, image2), f, tiles=chunks, margins=8)

    assert result1.ndim == ndim + 1
    assert result2.ndim == ndim + 1

    assert result1.shape[:-1] == result2.shape[:-1]
    assert result1.shape[-1] == 2
    assert result2.shape[-1] == 3

    result1 -= (0, 1)  # expected stats from uniform distribution
    result1 = Backend.to_numpy(result1)
    error = np.linalg.norm(result1.ravel(), ord=1) / result1.size
    aprint(f"Error = {error}")
    assert error < 0.001

    result2 -= (1, 0.5, 0)  # expected stats from uniform distribution
    result2 = Backend.to_numpy(result2)
    error = np.linalg.norm(result2.ravel(), ord=1) / result2.size
    aprint(f"Error = {error}")
    assert error < 0.001
