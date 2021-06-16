import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.scatter_gather_i2v import scatter_gather_i2v
from dexp.utils.timeit import timeit


def test_scatter_gather_i2v_numpy():
    with NumpyBackend():
        _test_scatter_gather_i2v()


def test_scatter_gather_i2v_cupy():
    try:
        with CupyBackend():
            _test_scatter_gather_i2v(length_xy=512, splits=4, filter_size=7)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_scatter_gather_i2v(ndim=3, length_xy=128, splits=4, filter_size=7):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    image1 = numpy.random.uniform(0, 1, size=(length_xy,) * ndim)
    image2 = numpy.random.uniform(0, 1, size=(length_xy,) * ndim)

    def f(x, y):
        return xp.stack([x.min(), x.max()]), xp.stack([y.max(), y.mean(), y.min()])

    try:
        with timeit("f"):
            result_ref_1, result_ref_2 = Backend.to_numpy(f(Backend.to_backend(image1), Backend.to_backend(image2)))
    except:
        print("Can't run this, not enough GPU memory!")
        result_ref_1 = 0
        result_ref_2 = 0

    with timeit("scatter_gather(f)"):
        chunks = (length_xy // splits,) * ndim
        result1, result2 = scatter_gather_i2v(f, (image1, image2), tiles=chunks, margins=8)

    print(result1.shape)
    print(result2.shape)

    assert result1.ndim == ndim + 1
    assert result2.ndim == ndim + 1

    assert result1.shape[-1] == 2
    assert result2.shape[-1] == 3

    mean = result1.mean(axis=tuple(a for a in range(ndim)))
    result1 -= mean
    result1 = Backend.to_numpy(result1)
    error = numpy.linalg.norm(result1.ravel(), ord=1) / result1.size
    print(f"Error = {error}")
    assert error < 0.001

    mean = result2.mean(axis=tuple(a for a in range(ndim)))
    result2 -= mean
    result2 = Backend.to_numpy(result2)
    error = numpy.linalg.norm(result2.ravel(), ord=1) / result2.size
    print(f"Error = {error}")
    assert error < 0.001

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(result, name='result', color=False)
    #
