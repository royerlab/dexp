from arbol import asection
from skimage.data import astronaut, logo

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.render.insert import insert_image


def demo_insert_numpy():
    with NumpyBackend():
        demo_insert()


def demo_insert_cupy():
    try:
        with CupyBackend():
            demo_insert()
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_insert(display=True):
    xp = Backend.get_xp_module()

    image_u = astronaut()
    image_v = logo()[::4, ::4, 0:3]

    with asection("insert_bl"):
        insert_bl = insert_image(image=image_u,
                                 inset_image=image_v,
                                 position='bottom_left',
                                 blend_mode='max',
                                 alpha=1,
                                 rgb=True)

    with asection("insert_tr_s2"):
        insert_tr_s2 = insert_image(image=image_u,
                                    inset_image=image_v,
                                    position='top_right',
                                    scale=2,
                                    blend_mode='max',
                                    alpha=1,
                                    rgb=True)

    with asection("insert_tr_s1h_add"):
        insert_tr_s1h_add = insert_image(image=image_u,
                                         inset_image=image_v,
                                         position='top_right',
                                         scale=1.5,
                                         blend_mode='add',
                                         alpha=1,
                                         rgb=True)

    if display:
        from napari import Viewer, gui_qt
        with gui_qt():
            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image_u), name='image_u')
            viewer.add_image(_c(image_v), name='image_v')
            viewer.add_image(_c(insert_bl), name='insert_bl')
            viewer.add_image(_c(insert_tr_s2), name='insert_tr_s2')
            viewer.add_image(_c(insert_tr_s1h_add), name='insert_tr_s1h_add')

            viewer.grid.enabled = True


if __name__ == "__main__":
    # demo_blend_cupy()
    demo_insert_numpy()
