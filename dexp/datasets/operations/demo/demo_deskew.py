import tempfile
from pathlib import Path

from arbol import aprint, asection

from dexp.datasets import ZDataset
from dexp.datasets.operations.deskew import dataset_deskew
from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.deskew import skew
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_deskew_numpy():
    with NumpyBackend():
        _demo_deskew()


def demo_deskew_cupy():
    try:
        with CupyBackend():
            _demo_deskew(length=128, zoom=4)
            return True
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def _demo_deskew(length=96, zoom=1, shift=1, angle=45, display=True):
    xp = Backend.get_xp_module()

    with asection("prepare simulated timelapse:"):
        # generate nuclei image:
        _, _, image = generate_nuclei_background_data(
            add_noise=False,
            length_xy=length,
            length_z_factor=1,
            independent_haze=True,
            sphere=True,
            zoom=zoom,
            add_offset=False,
            background_strength=0.07,
            dtype=xp.float32,
        )

        skewed = skew(image, nikon16x08na(), shift=shift, angle=angle, zoom=zoom, axis=0)
        # Single timepoint:
        skewed = skewed[xp.newaxis, ...]

    with tempfile.TemporaryDirectory() as tmpdir:
        aprint("created temporary directory", tmpdir)
        tmpdir = Path(tmpdir)

        with asection("Prepare dataset..."):
            input_path = tmpdir / "dataset.zarr"
            input_dataset = ZDataset(path=input_path, mode="w", store="dir")

            input_dataset.add_channel(name="channel", shape=skewed.shape, chunks=(1, 64, 64, 64), dtype=skewed.dtype)

            input_dataset.write_array(channel="channel", array=Backend.to_numpy(skewed))

            source_array = input_dataset.get_array("channel")

        with asection("Deskew..."):
            # output_folder:
            output_path = tmpdir / "deskewed.zarr"
            output_dataset = ZDataset(path=output_path, mode="w")

            # deskew:
            dataset_deskew(
                input_dataset=input_dataset,
                output_dataset=output_dataset,
                channels=("channel",),
                # slicing=(slice(0, 1),),
                mode="yang",
                dx=1,
                dz=1,
                angle=angle,
                flips=(False,),
                camera_orientation=0,
                depth_axis=0,
                lateral_axis=1,
                padding=False,
                devices=[
                    0,
                ],
            )

            output_dataset = ZDataset(path=output_path, mode="r")
            deskew_array = output_dataset.get_array("channel")

            assert deskew_array.shape[0] == 1

        if display:

            def _c(array):
                return Backend.to_numpy(array)

            import napari

            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(_c(source_array), name="source_array")
            viewer.add_image(_c(deskew_array), name="deconv_array")
            viewer.grid.enabled = True
            napari.run()


if __name__ == "__main__":
    if not demo_deskew_cupy():
        demo_deskew_numpy()
