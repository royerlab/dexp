from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.restoration.dehazing import dehaze
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def demo_dehaze_numpy():
    backend = NumpyBackend()
    demo_dehaze_data(backend)


def demo_dehaze_cupy():
    try:
        backend = CupyBackend()
        demo_dehaze_data(backend)
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_dehaze_data(backend, length_xy=320):
    xp = backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(backend,
                                                                      add_noise=True,
                                                                      length_xy=length_xy,
                                                                      length_z_factor=4,
                                                                      independent_haze=True)

    with timeit('dehaze'):
        dehazed = dehaze(backend, image, size=25)

    background_voxels_image = (1 - image_gt) * image
    background_voxels_dehazed = (1 - image_gt) * dehazed
    total_haze = xp.sum(background_voxels_image)
    total_remaining_haze = xp.sum(background_voxels_dehazed)

    percent_removed = (total_haze - total_remaining_haze) / total_haze

    print(f"percent_removed = {percent_removed}")

    from napari import gui_qt, Viewer
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image_gt), name='image_gt')
        viewer.add_image(_c(background), name='background')
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(dehazed), name='dehazed')


demo_dehaze_cupy()
demo_dehaze_numpy()
