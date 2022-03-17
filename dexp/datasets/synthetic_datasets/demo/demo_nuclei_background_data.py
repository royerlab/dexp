from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
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
        image_gt, background, image = generate_nuclei_background_data(
            add_noise=True, length_xy=length_xy, length_z_factor=1
        )

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_gt), name="image_gt")
        viewer.add_image(_c(background), name="restoration")
        viewer.add_image(_c(image), name="image")


if __name__ == "__main__":
    demo_nuclei_background_data_cupy()
    demo_nuclei_background_data_numpy()
