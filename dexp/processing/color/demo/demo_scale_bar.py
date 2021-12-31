from arbol import asection
from skimage.color import gray2rgba
from skimage.data import camera

from dexp.processing.color.scale_bar import insert_scale_bar
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_scale_bar_numpy():
    with NumpyBackend():
        demo_scale_bar()


def demo_scale_bar_cupy():
    try:
        with CupyBackend():
            demo_scale_bar()
        return True
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")
        return False


def demo_scale_bar(display=True):

    with asection("generate data"):
        image = Backend.to_backend(gray2rgba(camera()))

    with asection("Apply scale bar..."):
        image_with_scale_bar_br, _ = insert_scale_bar(
            image,
            length_in_unit=50,
            pixel_scale=0.406,
            translation="bottom_right",
            color=(1, 1, 1, 1),
        )
    with asection("Apply scale bar..."):
        image_with_scale_bar_tl, _ = insert_scale_bar(
            image,
            length_in_unit=50,
            pixel_scale=0.406,
            translation="top_left",
            color=(1, 1, 1, 1),
            font_size=24,
        )

    with asection("Apply scale bar..."):
        image_with_scale_bar_tr, _ = insert_scale_bar(
            image,
            length_in_unit=50,
            pixel_scale=0.406,
            translation="top_right",
            color=(1, 0.9, 1, 1),
            mode="add",
        )
    with asection("Apply scale bar..."):
        image_with_scale_bar_bl, _ = insert_scale_bar(
            image,
            length_in_unit=50,
            pixel_scale=0.406,
            translation="bottom_left",
            color=(1, 0.9, 1, 1),
        )

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image_with_scale_bar_br), name="image_with_scale_bar_br", rgb=True)
            viewer.add_image(_c(image_with_scale_bar_tl), name="image_with_scale_bar_tl", rgb=True)
            viewer.add_image(_c(image_with_scale_bar_tr), name="image_with_scale_bar_tr", rgb=True)
            viewer.add_image(_c(image_with_scale_bar_bl), name="image_with_scale_bar_bl", rgb=True)
            viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_scale_bar_cupy():
        demo_scale_bar_numpy()
