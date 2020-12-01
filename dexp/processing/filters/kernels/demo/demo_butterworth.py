from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.kernels.butterworth import butterworth_kernel


def demo_butterworth_numpy():
    with NumpyBackend():
        _demo_butterworth()


def demo_butterworth_cupy():
    try:
        with CupyBackend():
            _demo_butterworth()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_butterworth():
    xp = Backend.get_xp_module()

    b = butterworth_kernel(shape=(31, 31),
                           cutoffs=0.75,
                           cutoffs_in_freq_units=False,
                           epsilon=1,
                           order=7)

    b_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(b))))

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(b), name='b', colormap='viridis')
        viewer.add_image(_c(b_f), name='b_f', colormap='viridis')
        viewer.grid.enabled = True
        viewer.grid.shape = (2, 2)


demo_butterworth_cupy()
demo_butterworth_numpy()
