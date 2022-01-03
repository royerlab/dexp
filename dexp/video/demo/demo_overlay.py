import os
import random
import tempfile
from os.path import join

from arbol import aprint, asection
from dask.array.image import imread
from skimage.color import gray2rgba
from skimage.data import camera

from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.video.overlay import add_overlays_image_sequence


def demo_overlay_numpy():
    with NumpyBackend():
        demo_overlay(display=True)


def demo_overlay_cupy():
    try:
        with CupyBackend():
            demo_overlay(n=64, display=True)
            return True

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def demo_overlay(n=16, display=True):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    with asection("generate data"):
        image = gray2rgba(camera())

        # generate reference 'ground truth' timelapse
        images = (Backend.to_backend(image.copy()) for _ in range(n))

        # modify each image:
        images = (sp.ndimage.shift(image, shift=(random.uniform(-1, 2), random.uniform(-2, 1), 0)) for image in images)

        # Convert back images to 8 bit:
        images = list(image.astype(xp.uint8) for image in images)

    with tempfile.TemporaryDirectory() as tmpdir:
        aprint("created temporary directory", tmpdir)

        with asection("Save image sequences..."):
            # Create two subfolders for the two image sequences:
            input_folder = join(tmpdir, "input_images")
            os.makedirs(input_folder)

            # Save images from first sequence:
            for i, image_u in enumerate(images):
                from PIL import Image

                im = Image.fromarray(Backend.to_numpy(image_u))
                im.save(join(input_folder, f"frame_{i:05}.png"))

        with asection("Overlay..."):
            # Create output folder:
            output_folder = join(tmpdir, "overlay")
            os.makedirs(output_folder)

            add_overlays_image_sequence(
                input_path=input_folder,
                output_path=output_folder,
                scale_bar_length_in_unit=100,
                time_stamp_time_interval=20 / 60.0,
                margin=1,
            )

        # load images into dask arrays:
        input_images = imread(os.path.join(input_folder, "frame_*.png"))
        output_images = imread(os.path.join(output_folder, "frame_*.png"))

        if display:
            from napari import Viewer, gui_qt

            with gui_qt():

                def _c(array):
                    return Backend.to_numpy(array)

                viewer = Viewer(ndisplay=2)
                # viewer.add_image(_c(images), name='images', rgb=True)
                viewer.add_image(_c(input_images), name="input_images", rgb=True)
                viewer.add_image(_c(output_images), name="output_images", rgb=True)

                viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_overlay_numpy():
        demo_overlay_cupy()
