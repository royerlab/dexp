from arbol import asection

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.color.colormap import rgb_colormap
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_colormap_numpy():
    with NumpyBackend():
        demo_colormap()


def demo_colormap_cupy():
    try:
        with CupyBackend():
            demo_colormap(length_xy=512)
        return True
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")
        return False


def demo_colormap(length_xy=120, display=True):
    with asection("generate data"):
        _, _, image = generate_nuclei_background_data(
            add_noise=False, length_xy=length_xy, length_z_factor=1, background_strength=0.001, sphere=True, zoom=2
        )

        image -= image.min()
        image /= image.max()

    with asection("rgb_image"):
        rgb_image = rgb_colormap(image, cmap="turbo")

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name="image")
            viewer.add_image(_c(rgb_image), name="rgb_image")


if __name__ == "__main__":
    if not demo_colormap_cupy():
        demo_colormap_numpy()
