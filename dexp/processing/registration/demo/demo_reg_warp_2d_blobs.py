import numpy
from scipy.ndimage import gaussian_filter
from skimage.data import binary_blobs

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.processing.registration.reg_warp_nd import register_warp_nd
from dexp.utils.timeit import timeit


def demo_register_warp_2D_numpy():
    backend = NumpyBackend()
    register_warp_2D(backend)


def demo_register_warp_2D_cupy():
    try:
        backend = CupyBackend()
        register_warp_2D(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def register_warp_2D(backend, length_xy=512, warp_grid_size=4, reg_grid_size=8):
    image = binary_blobs(length=length_xy, seed=1, n_dim=2, blob_size_fraction=0.01, volume_fraction=0.05)
    image = image.astype(numpy.float32)
    image = gaussian_filter(image, sigma=4)

    magnitude = 10
    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(warp_grid_size,) * 2 + (2,))
    # vector_field[:, :, 0] = 10
    # vector_field[:, :, 1] = -3

    print(f"vector field applied: {vector_field}")

    with timeit("warp"):
        warped = warp(backend, image, vector_field, vector_field_zoom=4)

    chunks = tuple(s // reg_grid_size for s in image.shape)
    margins = tuple(c // 2 for c in chunks)

    unwarped = warped
    # for i in range(1):
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
        viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive')
        viewer.add_image(_c(warped), name='warped', colormap='bop blue', blending='additive', visible=False)
        viewer.add_image(_c(unwarped), name='unwarped', colormap='bop purple', blending='additive')


demo_register_warp_2D_cupy()
# demo_register_warp_2D_numpy()
