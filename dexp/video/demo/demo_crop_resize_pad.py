import os
import random
import tempfile
from os.path import join

import numpy
from arbol import aprint, asection
from dask.array.image import imread
from skimage.data import astronaut

from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.video.crop_resize_pad import crop_resize_pad_image_sequence


def demo_crop_resize_pad_numpy():
    with NumpyBackend():
        demo_video_crop_resize_pad(display=True)


def demo_crop_resize_pad_cupy():
    try:
        with CupyBackend():
            demo_video_crop_resize_pad(display=True)
            return True

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def demo_video_crop_resize_pad(n=64, display=True):
    sp = Backend.get_sp_module()

    with asection("Prepare dataset..."):
        image = Backend.to_backend(astronaut()[0:500, 0:500])

        # generate reference 'ground truth' timelapse
        images = (image.copy() for _ in range(n))

        # modify each image:
        images = (sp.ndimage.shift(image, shift=(random.uniform(-1, 2), random.uniform(-2, 1), 0)) for image in images)

        # Convert back images to 8 bit:
        images = list(image.astype(numpy.uint8) for image in images)

    with tempfile.TemporaryDirectory() as tmpdir:
        aprint("created temporary directory", tmpdir)

        with asection("Save image sequences..."):
            # Create two subfolders for the two image sequences:
            folder = join(tmpdir, "images")
            os.makedirs(folder)

            # Save images from sequence:
            for i, image in enumerate(images):
                from PIL import Image

                im = Image.fromarray(Backend.to_numpy(image))
                im.save(join(folder, f"frame_{i:05}.png"))

        with asection("resize sequence..."):
            # Create output folder:
            output_folder = join(tmpdir, "resized")
            os.makedirs(output_folder)

            # Perform blend:
            crop_resize_pad_image_sequence(
                input_path=folder, output_path=output_folder, crop=3, resize=(500, 400), pad_width=(3, 7)
            )

        # load images into dask arrays:
        images = imread(os.path.join(folder, "frame_*.png"))
        images_resized = imread(os.path.join(output_folder, "frame_*.png"))

        if display:
            from napari import Viewer, gui_qt

            with gui_qt():

                def _c(array):
                    return Backend.to_numpy(array)

                viewer = Viewer(ndisplay=2)
                viewer.add_image(_c(images), name="images_u", rgb=True)
                viewer.add_image(_c(images_resized), name="images_v", rgb=True)

                viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_crop_resize_pad_cupy():
        demo_crop_resize_pad_numpy()
