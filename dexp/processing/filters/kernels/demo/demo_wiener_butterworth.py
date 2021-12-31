import numpy

from dexp.processing.filters.kernels.gaussian import gaussian_kernel_nd
from dexp.processing.filters.kernels.wiener_butterworth import wiener_butterworth_kernel
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_wiener_butterworth_numpy():
    with NumpyBackend():
        _demo_wiener_butterworth()


def demo_wiener_butterworth_cupy():
    try:
        with CupyBackend():
            _demo_wiener_butterworth()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_wiener_butterworth():
    xp = Backend.get_xp_module()

    psf = gaussian_kernel_nd(size=31, ndim=2, dtype=numpy.float32)

    wb = wiener_butterworth_kernel(psf, cutoffs=None)

    psf_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(psf))))
    wb_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(wb))))

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(psf), name="psf")
        viewer.add_image(_c(wb), name="wb")
        viewer.add_image(_c(psf_f), name="psf_f", colormap="viridis")
        viewer.add_image(_c(wb_f), name="wb_f", colormap="viridis")
        viewer.grid.enabled = True
        viewer.grid.view = (2, 2)


demo_wiener_butterworth_cupy()
demo_wiener_butterworth_numpy()
