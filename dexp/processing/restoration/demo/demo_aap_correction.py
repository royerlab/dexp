from scipy.ndimage import gaussian_filter
from skimage.data import binary_blobs

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.restoration.aap_correction import axis_aligned_pattern_correction
from dexp.processing.restoration.test.test_aap_correction import add_patterned_noise


def demo_aap_correction_numpy():
    backend = NumpyBackend()
    demo_aap_correction(backend)


def demo_aap_correction_cupy():
    try:
        backend = CupyBackend()
        demo_aap_correction(backend)
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_aap_correction(backend: Backend, length_xy=128, level=0.3):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    image = binary_blobs(length=length_xy, seed=1, n_dim=3, volume_fraction=0.01)
    image = image.astype(xp.float32)
    image = gaussian_filter(image, sigma=4)
    image = add_patterned_noise(image, length_xy)

    corrected = axis_aligned_pattern_correction(backend,
                                                image,
                                                in_place=False)

    import napari

    with napari.gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = napari.Viewer()
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(corrected), name='corrected')


if __name__ == "__main__":
    demo_aap_correction_cupy()
    demo_aap_correction_numpy()
