from arbol import asection
from skimage.data import astronaut, logo

from dexp.processing.color.insert import insert_color_image
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_insert_numpy():
    with NumpyBackend():
        demo_insert()


def demo_insert_cupy():
    try:
        with CupyBackend():
            demo_insert()
        return True
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")
        return False


def demo_insert(display=True):
    image_u = Backend.to_backend(astronaut())
    image_v = Backend.to_backend(logo()[::4, ::4, 0:3])

    with asection("insert_bl"):
        insert_bl = insert_color_image(
            image=image_u, inset_image=image_v, translation="bottom_left", mode="max", alpha=1
        )

    with asection("insert_tr_s2"):
        insert_tr_s2 = insert_color_image(
            image=image_u,
            inset_image=image_v,
            translation="top_right",
            scale=2,
            mode="max",
            border_color=(1, 0, 0, 1),
            border_width=3,
            alpha=1,
        )

    with asection("insert_tr_s1h_add"):
        insert_tr_s1h_add = insert_color_image(
            image=image_u,
            inset_image=image_v,
            translation="top_right",
            scale=1.5,
            mode="add",
            border_color=(0, 0, 1, 1),
            border_width=10,
            alpha=1,
        )

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image_u), name="image_u")
            viewer.add_image(_c(image_v), name="image_v")
            viewer.add_image(_c(insert_bl), name="insert_bl")
            viewer.add_image(_c(insert_tr_s2), name="insert_tr_s2")
            viewer.add_image(_c(insert_tr_s1h_add), name="insert_tr_s1h_add")

            viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_insert_cupy():
        demo_insert_numpy()
