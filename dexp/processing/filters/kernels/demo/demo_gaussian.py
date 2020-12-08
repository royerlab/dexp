from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.kernels.gaussian import gaussian_kernel_nd


def demo_kernels_numpy():
    with NumpyBackend():
        _demo_kernels()


def demo_kernels_cupy():
    try:
        with CupyBackend():
            _demo_kernels()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_kernels():
    gaussian_kernel = gaussian_kernel_nd(ndim=4, size=17, sigma=2)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(gaussian_kernel), name='gaussian_kernel')


demo_kernels_cupy()
demo_kernels_numpy()
