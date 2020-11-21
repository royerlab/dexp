from napari import gui_qt, Viewer

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def demo_clean_dark_regions_numpy():
    backend = NumpyBackend()
    demo_clean_dark_regions_data(backend)


def demo_clean_dark_regions_cupy():
    # try:
    backend = CupyBackend()
    demo_clean_dark_regions_data(backend)
    # except (ModuleNotFoundError, NotImplementedError):
    #    print("Cupy module not found! ignored!")


def demo_clean_dark_regions_data(backend, length_xy=256):
    xp = backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(backend,
                                                                      add_noise=True,
                                                                      length_xy=length_xy,
                                                                      length_z_factor=1,
                                                                      independent_haze=False,
                                                                      background_stength=0.1)

    # remove zero level
    image = xp.clip(image - 95, 0, None)

    with timeit('dehaze_new'):
        cleaned = clean_dark_regions(backend, image, size=5, threshold=10, in_place=False)

    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_gt), name='image_gt')
        viewer.add_image(_c(background), name='background')
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(cleaned), name='cleaned', gamma=0.1)


demo_clean_dark_regions_cupy()
demo_clean_dark_regions_numpy()
