import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.scatter_gather import scatter_gather
from dexp.utils.timeit import timeit


def test_scatter_gather_numpy():
    backend = NumpyBackend()
    _test_scatter_gather(backend)


def test_scatter_gather_cupy():
    try:
        backend = CupyBackend()
        _test_scatter_gather(backend, length_xy=768, splits=4, filter_size=7)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_scatter_gather(backend, ndim=3, length_xy=128, splits=4, filter_size=7):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    image = numpy.random.uniform(0, 1, size=(length_xy,) * ndim)

    def f(x):
        return sp.ndimage.filters.uniform_filter(x, size=filter_size)

    try:
        with timeit("f"):
            result_ref = backend.to_numpy(f(backend.to_backend(image)))
    except:
        print("Can't run this, not enough GPU memory!")
        result_ref = 0 * image

    with timeit("scatter_gather(f)"):
        result = scatter_gather(backend, f, image, chunks=(length_xy // splits,) * ndim, margins=filter_size // 2)

    image = backend.to_numpy(image)
    result_ref = backend.to_numpy(result_ref)
    result = backend.to_numpy(result)

    error = numpy.linalg.norm(result.ravel() - result_ref.ravel(), ord=1) / result_ref.size
    print(f"Error = {error}")

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(result_ref, name='result_ref')
    #     viewer.add_image(result, name='result')

    assert error < 0.0001
