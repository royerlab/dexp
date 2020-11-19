import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def demo_warp_3d_numpy():
    try:
        backend = NumpyBackend()
        _demo_warp_3d(backend)
    except NotImplementedError:
        print("Numpy version not yet implemented")


def demo_warp_3d_cupy():
    try:
        backend = CupyBackend()
        _demo_warp_3d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_warp_3d(backend, length_xy=256, grid_size=8):
    with timeit("generate data"):
        _, _, image = generate_nuclei_background_data(backend,
                                                      add_noise=True,
                                                      length_xy=length_xy,
                                                      length_z_factor=1,
                                                      zoom=2)

    vector_field = numpy.random.uniform(low=-15, high=+15, size=(grid_size,)*3+(3,))

    with timeit("warp"):
        warped = warp(backend, image, vector_field)

    with timeit("dewarp"):
        dewarp = warp(backend, warped, -vector_field)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive', rendering='attenuated_mip')
        viewer.add_image(_c(warped), name='warped', colormap='bop purple', blending='additive', rendering='attenuated_mip')
        viewer.add_image(_c(dewarp), name='dewarp', colormap='bop blue', blending='additive', rendering='attenuated_mip')
        viewer.camera.ndisplay=3


demo_warp_3d_cupy()
demo_warp_3d_numpy()
