import os
import random
import tempfile
from os.path import join

import numpy
from arbol import aprint, asection
from dask.array.image import imread
from PIL import Image
from skimage.color import gray2rgba
from skimage.data import astronaut, camera, logo

from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.video.blend import blend_color_image_sequences


def demo_blend_numpy():
    with NumpyBackend():
        demo_blend(
            alphas=(0.1, 1, 0.9), scales=(1.0, 0.5, 0.3), translations=((0, 0), (100, 100), (50, 150)), display=True
        )
        demo_blend(display=True)


def demo_blend_cupy():
    try:
        with CupyBackend():
            demo_blend(
                alphas=(0.5, 1, 0.9), scales=(1.0, 0.5, 0.3), translations=((0, 0), (100, 100), (50, 150)), display=True
            )
            demo_blend(display=True)

            return True

    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def demo_blend(n=8, alphas=(1, 1, 0.9), scales=None, translations=None, display=True):
    sp = Backend.get_sp_module()

    with asection("Prepare dataset..."):
        image_u = Backend.to_backend(logo())
        image_v = Backend.to_backend(astronaut()[0:500, 0:500])
        image_w = Backend.to_backend(gray2rgba(camera()[0:500, 0:500]))

        # modulate alpha channel:
        image_u[:, 0:256, 3] = 128

        # generate reference 'ground truth' timelapse
        images_u = (image_u.copy() for _ in range(n))
        images_v = (image_v.copy() for _ in range(n))

        # modify each image:
        images_u = (
            sp.ndimage.shift(image, shift=(random.uniform(-1, 2), random.uniform(-2, 1), 0)) for image in images_u
        )
        images_v = (
            sp.ndimage.shift(image, shift=(random.uniform(-2, 1), random.uniform(-2, 3), 0)) for image in images_v
        )

        # Convert back images to 8 bit:
        images_u = list(image.astype(numpy.uint8) for image in images_u)
        images_v = list(image.astype(numpy.uint8) for image in images_v)

    with tempfile.TemporaryDirectory() as tmpdir:
        aprint("created temporary directory", tmpdir)

        with asection("Save image sequences..."):
            # Create two subfolders for the two image sequences:
            folder_u = join(tmpdir, "images_u")
            os.makedirs(folder_u)
            folder_v = join(tmpdir, "images_v")
            os.makedirs(folder_v)

            # Save images from first sequence:

            for i, image_u in enumerate(images_u):

                im = Image.fromarray(Backend.to_numpy(image_u))
                im.save(join(folder_u, f"frame_{i:05}.png"))

            # Save images from second sequence:
            for i, image_v in enumerate(images_v):

                im = Image.fromarray(Backend.to_numpy(image_v))
                im.save(join(folder_v, f"frame_{i:05}.png"))

            # Create png files for image w image_w

            im = Image.fromarray(Backend.to_numpy(image_w))
            path_w = join(tmpdir, "someimage.png")
            im.save(path_w)

        with asection("Blend sequences..."):
            # Create output folder:
            output_folder = join(tmpdir, "blend")
            os.makedirs(output_folder)

            # Perform blend:
            blend_color_image_sequences(
                input_paths=(folder_u, folder_v, path_w),
                output_path=output_folder,
                modes=("max", "mean", "max"),
                alphas=alphas,
                scales=scales,
                translations=translations,
            )

        # load images into dask arrays:
        images_u = imread(os.path.join(folder_u, "frame_*.png"))
        images_v = imread(os.path.join(folder_v, "frame_*.png"))
        images_w = imread(path_w)
        images_blend = imread(os.path.join(output_folder, "frame_*.png"))

        if display:
            from napari import Viewer, gui_qt

            with gui_qt():

                def _c(array):
                    return Backend.to_numpy(array)

                viewer = Viewer(ndisplay=2)
                viewer.add_image(_c(images_u), name="images_u", rgb=True)
                viewer.add_image(_c(images_v), name="images_v", rgb=True)
                viewer.add_image(_c(images_w), name="images_w", rgb=True)
                viewer.add_image(_c(images_blend), name="images_blend", rgb=True)

                viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_blend_cupy():
        demo_blend_numpy()
