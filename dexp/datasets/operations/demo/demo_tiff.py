import random
import tempfile
from pathlib import Path

from arbol import aprint, asection
from tifffile import imread

from dexp.datasets import ZDataset
from dexp.datasets.operations.tiff import dataset_tiff
from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_tiff_numpy():
    with NumpyBackend():
        _demo_tiff()


def demo_tiff_cupy():
    try:
        with CupyBackend():
            _demo_tiff(length_xy=128, zoom=2)
            return True
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def _demo_tiff(length_xy=96, zoom=1, n=16, display=True):
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
            sp.ndimage.shift(
                image, shift=(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))
            )
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
            dataset.set_slicing((slice(0, 4), ...))

            source_array = dataset.get_array("channel")

        with asection("Export to tiff..."):
            # output_folder:
            output_path = tmpdir / "file"

            # Do actual tiff export:
            dataset_tiff(dataset=dataset, dest_path=output_path, channels=("channel",))

            output_file = Path(str(output_path) + ".tiff")

            assert output_file.exists()

            tiff_image = imread(output_file)

            assert tiff_image.shape[0] == 4
            assert tiff_image.shape[1:] == source_array.shape[1:]

            # compute mean absolute errors:
            source_array = Backend.to_backend(source_array)
            tiff_image = Backend.to_backend(tiff_image)
            error = xp.mean(xp.absolute(source_array[0:4] - tiff_image))
            aprint(f"error = {error}")
            # Asserts that check if things behave as expected:
            assert error < 1e-6

            if display:

                def _c(array):
                    return Backend.to_numpy(array)

                import napari

                viewer = napari.Viewer(ndisplay=3)
                viewer.add_image(_c(source_array), name="source_array")
                viewer.add_image(_c(tiff_image), name="tiff_image")
                viewer.grid.enabled = True
                napari.run()

        with asection("Export to tiff, one file per timepoint..."):
            # output_folder:
            output_path = tmpdir / "projection"

            # Do actual tiff export:
            dataset_tiff(dataset=dataset, dest_path=output_path, channels=("channel",), project=0)

            output_file = Path(str(output_path) + ".tiff")

            assert output_file.exists()

            tiff_image = imread(output_file)

            assert tiff_image.shape[0] == 4
            assert tiff_image.ndim == 3
            assert tiff_image.shape[1:] == source_array.shape[2:]

            if display:

                def _c(array):
                    return Backend.to_numpy(array)

                import napari

                viewer = napari.Viewer(ndisplay=3)
                viewer.add_image(_c(source_array), name="source_array")
                viewer.add_image(_c(tiff_image), name="tiff_image")
                viewer.grid.enabled = True
                napari.run()


if __name__ == "__main__":
    if not demo_tiff_cupy():
        demo_tiff_numpy()
