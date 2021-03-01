from arbol import asection
from skimage.color import gray2rgba
from skimage.data import camera

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.color.border import add_border


def demo_border_numpy():
    with NumpyBackend():
        demo_border()


def demo_border_cupy():
    try:
        with CupyBackend():
            demo_border()
        return True
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")
        return False


def demo_border(display=True):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    with asection("generate data"):
        image = Backend.to_backend(gray2rgba(camera()))

    with asection("Add border..."):
        image_with_border = add_border(image,
                                       width=3,
                                       color=(1, 1, 1, 1)
                                       )

    with asection("Overlay border..."):
        image_with_border_overlayed = add_border(image,
                                                 width=3,
                                                 color=(1, 1, 1, 1),
                                                 over_image=True,
                                                 )

    if display:
        from napari import Viewer, gui_qt
        with gui_qt():
            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name='image', rgb=True)
            viewer.add_image(_c(image_with_border), name='image_with_border', rgb=True)
            viewer.add_image(_c(image_with_border_overlayed), name='image_with_border_overlayed', rgb=True)
            viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_border_cupy():
        demo_border_numpy()
