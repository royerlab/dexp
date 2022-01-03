from random import uniform

from arbol import asection
from skimage.color import gray2rgba
from skimage.data import camera

from dexp.processing.color.time_stamp import insert_time_stamp
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_time_stamp_numpy():
    with NumpyBackend():
        demo_time_stamp()


def demo_time_stamp_cupy():
    try:
        with CupyBackend():
            demo_time_stamp()
        return True
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")
        return False


def demo_time_stamp(n=32, display=True):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    with asection("generate data"):
        image = Backend.to_backend(gray2rgba(camera()))

        # generate reference 'ground truth' timelapse
        images = (image.copy() for _ in range(n))

        # modify each image:
        images = (sp.ndimage.shift(image, shift=(uniform(-1, 2), uniform(-2, 1), 0)) for image in images)

        # Convert back images to 8 bit:
        images = list(image.astype(xp.uint8) for image in images)

    with asection("Apply time stamp..."):
        images_with_time_stamps = []
        for tp, image in enumerate(images):
            image_with_time_stamp = insert_time_stamp(
                image=image,
                time_point_index=tp,
                nb_time_points=len(images),
                start_time=0,
                time_interval=20 / 60.0,
                unit="min",
                translation="top_right",
                color=(1, 1, 1, 1),
                alpha=1.0,
            )
            images_with_time_stamps.append(image_with_time_stamp)

        # Convert sequences to arrays:
        # images = xp.stack(images)
        images_with_time_stamp_tr = xp.stack(images_with_time_stamps)

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(images_with_time_stamp_tr), name="images_with_time_stamp_tr", rgb=True)
            viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_time_stamp_cupy():
        demo_time_stamp_numpy()
