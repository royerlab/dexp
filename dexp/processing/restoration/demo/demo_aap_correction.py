from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.restoration.aap_correction import axis_aligned_pattern_correction
from dexp.processing.synthetic_datasets.binary_blobs import binary_blobs


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

    image_blobs = binary_blobs(backend, length=length_xy, n_dim=3, blob_size_fraction=0.1, volume_fraction=0.5).astype('f4')
    image_blobs = sp.ndimage.gaussian_filter(image_blobs, sigma=4)
    image_blobs = image_blobs / xp.max(image_blobs)

    a = xp.random.uniform(0, 1, size=length_xy) ** 0.33
    b = xp.random.uniform(0, 1, size=length_xy) ** 0.33
    c = xp.random.uniform(0, 1, size=length_xy) ** 0.33

    axis_aligned_noise = xp.einsum('i,j,k', a, b, c)

    image = image_blobs.copy()
    image += level * axis_aligned_noise
    # image = random_noise(image, mode="s&p", amount=0.03, seed=0, clip=False)

    corrected = axis_aligned_pattern_correction(backend, image)

    import napari

    with napari.gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = napari.Viewer()
        viewer.add_image(_c(image_blobs), name='image_blobs')
        viewer.add_image(_c(axis_aligned_noise), name='axis_aligned_noise')
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(corrected), name='corrected')


demo_aap_correction_cupy()
demo_aap_correction_numpy()
