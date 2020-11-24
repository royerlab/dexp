import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.reg_warp_nd import register_warp_nd
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def demo_register_warp_nD_numpy():
    backend = NumpyBackend()
    register_warp_nD(backend)


def demo_register_warp_nD_cupy():
    try:
        backend = CupyBackend()
        register_warp_nD(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def register_warp_nD(backend, length_xy=256, warp_grid_size=2, reg_grid_size=6):
    _, _, image = generate_nuclei_background_data(backend,
                                                  add_noise=False,
                                                  length_xy=length_xy,
                                                  length_z_factor=1,
                                                  zoom=2)
    xp = backend.get_xp_module()

    magnitude = 10
    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,))
    # vector_field[:, :, 0] = 10
    # vector_field[:, :, 1] = -3

    print(f"vector field applied: {vector_field}")

    with timeit("warp"):
        warped = warp(backend, image, vector_field, vector_field_zoom=8)

    image += xp.random.uniform(-20, 20, size=warped.shape)
    warped += xp.random.uniform(-20, 20, size=warped.shape)

    chunks = tuple(s // reg_grid_size for s in image.shape)
    margins = tuple(c // 2 for c in chunks)

    unwarped = warped
    for i in range(1):
        with timeit("register_warp_nd"):
            model = register_warp_nd(backend, image, unwarped, chunks=chunks, margins=margins)
            print(f"vector field found: {vector_field}")

        with timeit("unwarp"):
            unwarped = warp(backend, unwarped, -model.vector_field, vector_field_zoom=4)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive', rendering='attenuated_mip')
        viewer.add_image(_c(warped), name='warped', colormap='bop blue', blending='additive', visible=False, rendering='attenuated_mip')
        viewer.add_image(_c(unwarped), name='unwarped', colormap='bop purple', blending='additive', rendering='attenuated_mip')


demo_register_warp_nD_cupy()
demo_register_warp_nD_numpy()
