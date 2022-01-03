from arbol import asection

from dexp.datasets.synthetic_datasets import binary_blobs
from dexp.processing.color.blend import blend_color_images
from dexp.processing.color.conversions import gray2rgba
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_blend_numpy():
    with NumpyBackend():
        demo_blend()


def demo_blend_cupy():
    try:
        with CupyBackend():
            demo_blend()
        return True
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")
        return False


def demo_blend(length_xy=512, display=True):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    image_u = binary_blobs(length=length_xy, n_dim=2, blob_size_fraction=0.5, volume_fraction=0.3).astype(xp.float32)
    image_v = binary_blobs(length=length_xy, n_dim=2, blob_size_fraction=0.5, volume_fraction=0.3).astype(xp.float32)

    image_u = sp.ndimage.gaussian_filter(image_u, sigma=15)
    image_u = 255 * image_u / xp.max(image_u)
    image_v = sp.ndimage.gaussian_filter(image_v, sigma=15)
    image_v = 255 * image_v / xp.max(image_v)

    # convert to 8 bit:
    image_u = image_u.astype(xp.uint8)
    image_v = image_v.astype(xp.uint8)

    # convert to RGBA:
    image_u = gray2rgba(image_u)
    image_v = gray2rgba(image_v)

    # Keep only one channel out of 3:
    image_u[..., 0] = 0
    image_u[..., 3] = image_u[..., 1]
    image_v[..., 1] = 0
    image_v[..., 3] = image_v[..., 0]

    # Move to backend:
    image_u = Backend.to_backend(image_u)
    image_v = Backend.to_backend(image_v)

    # modulate alpha channel:
    # image_u[:, 0:256, 3] = 128
    # image_v[0:256, :, 3] = 128

    with asection("blend_mean"):
        blend_mean = blend_color_images(images=(image_u, image_v), alphas=(1, 1), modes=("max", "mean"))

    with asection("blend_add"):
        blend_add = blend_color_images(images=(image_u, image_v), alphas=(1, 1), modes=("max", "add"))

    # with asection("blend_erfadd"):
    #     blend_erfadd = blend((image_u, image_v), (1, 0.5), mode='erfadd')

    with asection("blend_max"):
        blend_max = blend_color_images(images=(image_u, image_v), alphas=(1, 1), modes=("max", "max"))

    with asection("blend_over"):
        blend_over = blend_color_images(images=(image_u, image_v), alphas=(1, 1), modes=("max", "over"))

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image_u), name="image_u")
            viewer.add_image(_c(image_v), name="image_v")
            viewer.add_image(_c(blend_mean), name="blend_mean")
            viewer.add_image(_c(blend_add), name="blend_add")
            # viewer.add_image(_c(blend_erfadd), name='blend_erfadd')
            viewer.add_image(_c(blend_max), name="blend_max")
            viewer.add_image(_c(blend_over), name="blend_over")
            viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_blend_cupy():
        demo_blend_numpy()
