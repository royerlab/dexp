import random
import tempfile
from os.path import join

from arbol import aprint, asection

from dexp.datasets import ZDataset
from dexp.datasets.operations.copy import dataset_copy
from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_copy_numpy():
    with NumpyBackend():
        _demo_copy()


def demo_copy_cupy():
    try:
        with CupyBackend():
            _demo_copy(length_xy=128, zoom=2)
            return True
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def _demo_copy(length_xy=96, zoom=1, n=16, display=True):
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

        with asection("Prepare dataset..."):
            input_path = join(tmpdir, "dataset.zarr")
            dataset = ZDataset(path=input_path, mode="w", store="dir")

            dataset.add_channel(name="channel", shape=images.shape, chunks=(1, 64, 64, 64), dtype=images.dtype)

            dataset.write_array(channel="channel", array=Backend.to_numpy(images))

            source_array = dataset.get_array("channel")

        with asection("Copy..."):
            # output_folder:
            output_path = join(tmpdir, "copy.zarr")
            dataset.set_slicing((slice(0, 4),))
            copied_dataset = ZDataset(path=output_path, mode="w")

            # Do actual copy:
            dataset_copy(
                input_dataset=dataset,
                output_dataset=copied_dataset,
                channels=("channel",),
                workers=4,
                workersbackend="threading",
            )

            copied_array = copied_dataset.get_array("channel")

            assert copied_array.shape[0] == 4
            assert copied_array.shape[1:] == source_array.shape[1:]

        if display:

            def _c(array):
                return Backend.to_numpy(array)

            import napari

            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(_c(source_array), name="source_array")
            viewer.add_image(_c(copied_array), name="copied_array")
            viewer.grid.enabled = True
            napari.run()


if __name__ == "__main__":
    if not demo_copy_cupy():
        demo_copy_numpy()
