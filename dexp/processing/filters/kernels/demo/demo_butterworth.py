from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.kernels.butterworth import butterworth_kernel


def demo_butterworth_numpy():
    backend = NumpyBackend()
    _demo_butterworth(backend)


def demo_butterworth_cupy():
    try:
        backend = CupyBackend()
        _demo_butterworth(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_butterworth(backend):
    xp = backend.get_xp_module()

    b = butterworth_kernel(backend,
                           shape=(31, 31),
                           cutoffs=0.75,
                           cutoffs_in_freq_units=False,
                           epsilon=1,
                           order=7)

    b_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(b))))

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(b), name='b', colormap='viridis')
        viewer.add_image(_c(b_f), name='b_f', colormap='viridis')
        viewer.grid_view(2, 2, 1)


demo_butterworth_cupy()
demo_butterworth_numpy()
