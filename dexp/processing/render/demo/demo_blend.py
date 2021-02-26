from arbol import asection
from skimage.data import logo

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.render.blend import blend_images


def demo_blend_numpy():
    with NumpyBackend():
        demo_blend()


def demo_blend_cupy():
    try:
        with CupyBackend():
            demo_blend()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_blend(display=True):
    xp = Backend.get_xp_module()

    image_u = logo()
    image_v = xp.flip(logo(), axis=1)

    # modulate alpha channel:
    image_u[:, 0:256, 3] = 128
    image_v[0:256, :, 3] = 128

    with asection("blend_mean"):
        blend_mean = blend_images((image_u, image_v), (1, 1), mode='mean')

    with asection("blend_add"):
        blend_add = blend_images((image_u, image_v), (1, 1), mode='add')

    # with asection("blend_erfadd"):
    #     blend_erfadd = blend((image_u, image_v), (1, 0.5), mode='erfadd')

    with asection("blend_max"):
        blend_max = blend_images((image_u, image_v), (1, 1), mode='max')

    with asection("blend_alpha"):
        blend_alpha = blend_images((image_u, image_v), (1, 1), mode='alpha')

    if display:
        from napari import Viewer, gui_qt
        with gui_qt():
            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image_u), name='image_u')
            viewer.add_image(_c(image_v), name='image_v')
            viewer.add_image(_c(blend_mean), name='blend_mean')
            viewer.add_image(_c(blend_add), name='blend_add')
            # viewer.add_image(_c(blend_erfadd), name='blend_erfadd')
            viewer.add_image(_c(blend_max), name='blend_max')
            viewer.add_image(_c(blend_alpha), name='blend_alpha')
            viewer.grid.enabled = True


if __name__ == "__main__":
    # demo_blend_cupy()
    demo_blend_numpy()
