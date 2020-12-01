from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.synthetic_datasets.binary_blobs import binary_blobs
from dexp.utils.timeit import timeit


def demo_blobs_numpy():
    with NumpyBackend():
        demo_blobs()


def demo_blobs_cupy():
    try:
        with CupyBackend():
            demo_blobs()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_blobs(length_xy=320):
    with timeit("generate data"):
        image_blobs = binary_blobs(length=length_xy, n_dim=3, blob_size_fraction=0.07, volume_fraction=0.1).astype('f4')

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_blobs), name='image_gt')


if __name__ == "__main__":
    demo_blobs_cupy()
    demo_blobs_numpy()
