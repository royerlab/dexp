import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.kernels.gaussian import gaussian_kernel_nd
from dexp.processing.filters.kernels.wiener_butterworth import wiener_butterworth_kernel


def demo_wiener_butterworth_numpy():
    backend = NumpyBackend()
    _demo_wiener_butterworth(backend)


def demo_wiener_butterworth_cupy():
    try:
        backend = CupyBackend()
        _demo_wiener_butterworth(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_wiener_butterworth(backend):

    xp = backend.get_xp_module()

    psf = gaussian_kernel_nd(backend,
                             size=9,
                             ndim=2,
                             dtype=numpy.float32)
    psf_b = wiener_butterworth_kernel(backend, psf)

    psf_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(psf))))
    psf_b_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(psf_b))))


    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(psf), name='psf')
        viewer.add_image(_c(psf_b), name='psf_b')
        viewer.add_image(_c(psf_f), name='psf_f')
        viewer.add_image(_c(psf_b_f), name='psf_b_f')
        viewer.grid_view(2, 2, 1)


demo_wiener_butterworth_cupy()
demo_wiener_butterworth_numpy()
