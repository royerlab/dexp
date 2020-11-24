import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.kernels.butterworth import butterworth_kernel
from dexp.processing.filters.kernels.gaussian import gaussian_kernel_nd
from dexp.processing.filters.kernels.wiener import wiener_kernel
from dexp.processing.filters.kernels.wiener_butterworth import wiener_butterworth_kernel


def demo_wiener_numpy():
    backend = NumpyBackend()
    _demo_wiener(backend)


def demo_wiener_cupy():
    try:
        backend = CupyBackend()
        _demo_wiener(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_wiener(backend):

    xp = backend.get_xp_module()

    psf = gaussian_kernel_nd(backend,
                             size=31,
                             ndim=2,
                             dtype=numpy.float32)

    psf_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(psf))))

    w = wiener_kernel(backend, kernel=psf)

    w_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(w))))


    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(psf), name='psf')
        viewer.add_image(_c(psf_f), name='psf_f', colormap='viridis')
        viewer.add_image(_c(w), name='b')
        viewer.add_image(_c(w_f), name='b_f', colormap='viridis')
        viewer.grid_view(2, 2, 1)


demo_wiener_cupy()
demo_wiener_numpy()
