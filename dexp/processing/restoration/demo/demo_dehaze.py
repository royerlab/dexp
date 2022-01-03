import numpy

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.restoration.dehazing import dehaze
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_dehaze_numpy():
    with NumpyBackend():
        demo_dehaze_data()


def demo_dehaze_cupy():
    try:
        with CupyBackend():
            demo_dehaze_data()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_dehaze_data(length_xy=320):
    xp = Backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(
            add_noise=True, length_xy=length_xy, length_z_factor=4, independent_haze=True, dtype=numpy.float32
        )

    with timeit("dehaze"):
        dehazed = dehaze(image, size=25, in_place=False, correct_max_level=True)

    # compute percentage of haze removed:
    background_voxels_image = (1 - image_gt) * image
    background_voxels_dehazed = (1 - image_gt) * dehazed
    total_haze = xp.sum(background_voxels_image)
    total_remaining_haze = xp.sum(background_voxels_dehazed)
    percent_removed = (total_haze - total_remaining_haze) / total_haze
    print(f"percent_removed = {percent_removed}")

    # compute error on non-background voxels
    non_background_voxels_image = image_gt * image
    non_background_voxels_dehazed = image_gt * dehazed
    average_error = xp.mean(xp.absolute(non_background_voxels_image - non_background_voxels_dehazed))
    print(f"average_error = {average_error}")

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_gt), name="image_gt")
        viewer.add_image(_c(background), name="background")
        viewer.add_image(_c(image), name="image")
        viewer.add_image(_c(dehazed), name="dehazed")
        viewer.grid.enabled = True


if __name__ == "__main__":
    demo_dehaze_cupy()
    demo_dehaze_numpy()
