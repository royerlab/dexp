from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def demo_nuclei_background_data_numpy():
    backend = NumpyBackend()
    demo_nuclei_background_data(backend)


def demo_nuclei_background_data_cupy():
    try:
        backend = CupyBackend()
        demo_nuclei_background_data(backend)
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_nuclei_background_data(backend, length_xy=320):
    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(backend,
                                                                      add_noise=True,
                                                                      length_xy=length_xy,
                                                                      length_z_factor=4)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_gt), name='image_gt')
        viewer.add_image(_c(background), name='restoration')
        viewer.add_image(_c(image), name='image')


demo_nuclei_background_data_cupy()
demo_nuclei_background_data_numpy()
