import numpy
import scipy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.reg_warp_nd import register_warp_nd
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.utils.timeit import timeit


def demo_register_warp_3d_numpy():
    backend = NumpyBackend()
    _register_warp_3d(backend)


def demo_register_warp_3d_cupy():
    try:
        backend = CupyBackend()
        _register_warp_3d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _register_warp_3d(backend, length_xy=256, warp_grid_size=3, reg_grid_size=6, display=True):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    _, _, image = generate_nuclei_background_data(backend,
                                                  add_noise=False,
                                                  length_xy=length_xy,
                                                  length_z_factor=1,
                                                  independent_haze=True,
                                                  sphere=True,
                                                  zoom=2,
                                                  dtype=numpy.float32)

    image = image[0:length_xy * 2 - 3, 0:length_xy * 2 - 5, 0:length_xy * 2 - 7]

    with timeit("warp"):
        magnitude = 10
        vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 3 + (3,))
        warped = warp(backend, image, vector_field, vector_field_upsampling=8)
        print(f"vector field applied: {vector_field}")

    with timeit("add noise"):
        image += xp.random.uniform(0, 40, size=image.shape)
        warped += xp.random.uniform(0, 40, size=warped.shape)

    with timeit(f"register_warp_nd"):
        chunks = tuple(s // reg_grid_size for s in image.shape)
        margins = tuple(max(4, c // 3) for c in chunks)
        print(f"chunks={chunks}, margins={margins}")
        model = register_warp_nd(backend, image, warped, chunks=chunks, margins=margins)
        model.clean(backend)
        # print(f"vector field found: {vector_field}")

    with timeit("unwarp"):
        _, unwarped = model.apply(backend, image, warped, vector_field_upsampling=4)

    vector_field = scipy.ndimage.zoom(vector_field, zoom=(2, 2, 2, 1), order=1)

    if display:
        from napari import Viewer, gui_qt
        with gui_qt():
            def _c(array):
                return backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive', rendering='attenuated_mip', attenuation=0.01)
            viewer.add_image(_c(warped), name='warped', colormap='bop blue', blending='additive', visible=False, rendering='attenuated_mip', attenuation=0.01)
            viewer.add_image(_c(unwarped), name='unwarped', colormap='bop purple', blending='additive', rendering='attenuated_mip', attenuation=0.01)
            viewer.add_vectors(_c(vector_field), name='gt vector_field')
            viewer.add_vectors(_c(model.vector_field), name='model vector_field')

    return image, warped, unwarped, model


if __name__ == "__main__":
    demo_register_warp_3d_cupy()
    # demo_register_warp_3d_numpy()
