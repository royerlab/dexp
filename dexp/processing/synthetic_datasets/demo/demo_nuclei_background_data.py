from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def demo_nuclei_background_data_numpy():
    with NumpyBackend():
        demo_nuclei_background_data()


def demo_nuclei_background_data_cupy():
    try:
        with CupyBackend():
            demo_nuclei_background_data()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_nuclei_background_data(length_xy=320):
    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(add_noise=True,
                                                                      length_xy=length_xy,
                                                                      length_z_factor=4)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_gt), name='image_gt')
        viewer.add_image(_c(background), name='restoration')
        viewer.add_image(_c(image), name='image')


if __name__ == "__main__":
    demo_nuclei_background_data_cupy()
    demo_nuclei_background_data_numpy()
