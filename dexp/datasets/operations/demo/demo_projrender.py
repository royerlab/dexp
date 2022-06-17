import random
import tempfile
from pathlib import Path

from arbol import aprint, asection
from dask.array.image import imread

from dexp.datasets import ZDataset
from dexp.datasets.operations.projrender import dataset_projection_rendering
from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_projrender_numpy():
    with NumpyBackend():
        _demo_projrender()


def demo_projrender_cupy():
    try:
        with CupyBackend():
            _demo_projrender(length_xy=320, zoom=4)
            return True
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def _demo_projrender(length_xy=96, zoom=1, n=64, display=True):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # generate nuclei image:
    _, _, image = generate_nuclei_background_data(
        add_noise=False,
        length_xy=length_xy,
        length_z_factor=1,
        independent_haze=True,
        sphere=True,
        zoom=zoom,
        dtype=xp.float32,
    )

    with asection("prepare simulated timelapse:"):
        # move to backend:
        image = Backend.to_backend(image)

        # generate reference 'ground truth' timelapse
        images = [image.copy() for _ in range(n)]

        # modify each image:
        images = [
            sp.ndimage.shift(image, shift=(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))
            for image in images
        ]
        images = [image + random.uniform(-10, 10) for image in images]

        # turn into array:
        images = xp.stack(images)

    with tempfile.TemporaryDirectory() as tmpdir:
        aprint("created temporary directory", tmpdir)
        tmpdir = Path(tmpdir)

        with asection("Prepare dataset..."):
            input_path = tmpdir / "dataset.zarr"
            dataset = ZDataset(path=input_path, mode="w", store="dir")

            dataset.add_channel(name="channel", shape=images.shape, chunks=(1, 64, 64, 64), dtype=images.dtype)

            dataset.write_array(channel="channel", array=Backend.to_numpy(images))

        with asection("Project..."):
            # output_folder:
            output_path = tmpdir / "projections"

            # Do actual projection:
            dataset_projection_rendering(
                input_dataset=dataset,
                channels=("channel",),
                output_path=output_path,
                legend_size=0.05,
                devices=[0],
                overwrite=True,
            )

        # load images into a dask array:
        filename_pattern = output_path / "frame_*.png"
        images = imread(str(filename_pattern))

        if display:
            from napari import Viewer, gui_qt

            with gui_qt():

                def _c(array):
                    return Backend.to_numpy(array)

                viewer = Viewer(ndisplay=3)
                viewer.add_image(_c(images), name="images", rgb=True)
                viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_projrender_cupy():
        demo_projrender_numpy()
