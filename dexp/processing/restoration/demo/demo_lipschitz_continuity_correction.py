import numpy
from scipy.ndimage import median_filter
from skimage.data import camera
from skimage.util import random_noise

from dexp.processing.restoration.lipshitz_correction import (
    lipschitz_continuity_correction,
)
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_lipschitz_correction_numpy():
    with NumpyBackend():
        demo_lipschitz_correction()


def demo_lipschitz_correction_cupy():
    try:
        with CupyBackend():
            demo_lipschitz_correction()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_lipschitz_correction():
    image = camera().astype(numpy.float32) / 255
    image = random_noise(image, mode="gaussian", var=0.005, seed=0, clip=False)
    image = random_noise(image, mode="s&p", amount=0.03, seed=0, clip=False)

    corrected = lipschitz_continuity_correction(image, lipschitz=0.15, in_place=False)

    median = median_filter(image, size=3)

    import napari

    with napari.gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = napari.Viewer()
        viewer.add_image(_c(image), name="image")
        viewer.add_image(_c(median), name="median_filtered")
        viewer.add_image(_c(corrected), name="corrected")
        viewer.grid.enabled = True


if __name__ == "__main__":
    demo_lipschitz_correction_cupy()
    demo_lipschitz_correction_numpy()
