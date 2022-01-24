from napari import Viewer, gui_qt

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_clean_dark_regions_numpy():
    with NumpyBackend():
        demo_clean_dark_regions_data()


def demo_clean_dark_regions_cupy():
    try:
        with CupyBackend():
            demo_clean_dark_regions_data()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_clean_dark_regions_data(length_xy=256):
    xp = Backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(
            add_noise=True, length_xy=length_xy, length_z_factor=1, independent_haze=False, background_strength=0.2
        )

    # remove zero level
    image = xp.clip(image - 95, 0, None)

    with timeit("dehaze_new"):
        cleaned = clean_dark_regions(image, size=5, threshold=30, in_place=False)

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_gt), name="image_gt")
        viewer.add_image(_c(background), name="background")
        viewer.add_image(_c(image), name="image")
        viewer.add_image(_c(cleaned), name="cleaned", gamma=0.1)
        viewer.grid.enabled = True


if __name__ == "__main__":
    demo_clean_dark_regions_cupy()
    demo_clean_dark_regions_numpy()
