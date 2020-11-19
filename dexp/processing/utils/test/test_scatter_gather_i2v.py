import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.processing.utils.scatter_gather_i2v import scatter_gather_i2v
from dexp.utils.timeit import timeit


def test_scatter_gather_i2v_numpy():
    backend = NumpyBackend()
    _test_scatter_gather_i2v(backend)


def test_scatter_gather_i2v_cupy():
    try:
        backend = CupyBackend()
        _test_scatter_gather_i2v(backend, length_xy=512, splits=4, filter_size=7)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_scatter_gather_i2v(backend, ndim=3, length_xy=128, splits=4, filter_size=7):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    image = numpy.random.uniform(0, 1, size=(length_xy,) * ndim)

    def f(x):
        return xp.asarray([x.min(), x.max()])

    try:
        with timeit("f"):
            result_ref = backend.to_numpy(f(backend.to_backend(image)))
    except:
        print("Can't run this, not enough GPU memory!")
        result_ref = 0 * image + 17

    with timeit("scatter_gather(f)"):
        chunks = (length_xy // splits,) * ndim
        result = scatter_gather_i2v(backend, f, image, chunks=chunks, margins=8)

    print(result)

    assert result.ndim == ndim+1

    mean = result.mean(axis=tuple(a for a in range(ndim)))

    result -= mean

    result = backend.to_numpy(result)

    error = numpy.linalg.norm(result.ravel(), ord=1) / result.size
    print(f"Error = {error}")

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(result, name='result', rgb=False)
    #
    assert error < 0.0001
