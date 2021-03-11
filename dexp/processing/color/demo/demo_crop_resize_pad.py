from arbol import asection
from skimage.color import gray2rgba
from skimage.data import camera

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.color.border import add_border
from dexp.processing.color.crop_resize_pad import crop_resize_pad_color_image


def demo_crop_resize_pad_numpy():
    with NumpyBackend():
        demo_crop_resize_pad()


def demo_crop_resize_pad_cupy():
    try:
        with CupyBackend():
            demo_crop_resize_pad()
        return True
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")
        return False


def demo_crop_resize_pad(display=True):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    with asection("generate data"):
        image = Backend.to_backend(gray2rgba(camera()))

    # def crop_resize_pad_color_image(image,
    #                                 crop: Tuple[Tuple[int, int], ...] = None,
    #                                 resize: Tuple[int, ...] = None,
    #                                 resize_order: int = 3,
    #                                 resize_mode: str = 'constant',
    #                                 pad: Tuple[Tuple[int, int], ...] = None,
    #                                 pad_color: Tuple[float, float, float, float] = (0, 0, 0, 0),
    #                                 rgba_value_max: float = 255

    with asection("crop_resize_pad ..."):
        crop_resize_pad_image = crop_resize_pad_color_image(image,
                                                            crop=3,
                                                            resize=(500, 400),
                                                            pad_width=(3, 7)
                                                            )

        assert crop_resize_pad_image.shape == (500+10, 400+10, 4)


    if display:
        from napari import Viewer, gui_qt
        with gui_qt():
            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name='image', rgb=True)
            viewer.add_image(_c(crop_resize_pad_image), name='crop_resize_pad_image', rgb=True)
            viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_crop_resize_pad_cupy():
        demo_crop_resize_pad_numpy()
