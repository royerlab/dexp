import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.timeit import timeit


def test_scatter_gather_i2i_numpy():
    with NumpyBackend():
        _test_scatter_gather_i2i()


def test_scatter_gather_i2i_cupy():
    try:
        with CupyBackend():
            _test_scatter_gather_i2i(length_xy=512, splits=4, filter_size=7)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _test_scatter_gather_i2i(ndim=3, length_xy=128, splits=4, filter_size=7):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    image = numpy.random.uniform(0, 1, size=(length_xy,) * ndim)

    def f(x):
        return sp.ndimage.filters.uniform_filter(x, size=filter_size)

    try:
        with timeit("f"):
            result_ref = Backend.to_numpy(f(Backend.to_backend(image)))
    except:
        print("Can't run this, not enough GPU memory!")
        result_ref = 0 * image + 17

    with timeit("scatter_gather(f)"):
        result = scatter_gather_i2i(f, image, tiles=(length_xy // splits,) * ndim, margins=filter_size // 2)

    image = Backend.to_numpy(image)
    result_ref = Backend.to_numpy(result_ref)
    result = Backend.to_numpy(result)

    error = numpy.linalg.norm(result.ravel() - result_ref.ravel(), ord=1) / result_ref.size
    print(f"Error = {error}")

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image, name='image')
    #     viewer.add_image(result_ref, name='result_ref')
    #     viewer.add_image(result, name='result')

    assert error < 0.0001
