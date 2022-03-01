import numpy as np
from arbol import aprint

from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.backends import Backend
from dexp.utils.testing.testing import execute_both_backends
from dexp.utils.timeit import timeit


@execute_both_backends
def test_scatter_gather_i2i(ndim=3, length_xy=128, splits=4, filter_size=7):
    sp = Backend.get_sp_module()
    rng = np.random.default_rng()

    image = rng.uniform(0, 1, size=(length_xy,) * ndim)

    def f(x):
        return sp.ndimage.uniform_filter(x, size=filter_size)

    try:
        with timeit("f"):
            result_ref = Backend.to_numpy(f(Backend.to_backend(image)))
    except RuntimeError:
        print("Can't run this, not enough GPU memory!")
        result_ref = 0 * image + 17

    with timeit("scatter_gather(f)"):
        result = scatter_gather_i2i(f, image, tiles=(length_xy // splits,) * ndim, margins=filter_size // 2)

    image = Backend.to_numpy(image)
    result_ref = Backend.to_numpy(result_ref)
    result = Backend.to_numpy(result)

    error = np.linalg.norm(result.ravel() - result_ref.ravel(), ord=1) / result_ref.size
    aprint(f"Error = {error}")

    assert error < 0.0001
