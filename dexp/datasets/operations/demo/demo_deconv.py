import random
import tempfile
from pathlib import Path

from arbol import aprint, asection

from dexp.datasets import ZDataset
from dexp.datasets.operations.deconv import dataset_deconv
from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_deconv_numpy():
    with NumpyBackend():
        _demo_deconv()


def demo_deconv_cupy():
    try:
        with CupyBackend():
            _demo_deconv(length_xy=128, zoom=4)
            return True
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")
        return False


def _demo_deconv(length_xy=96, zoom=1, n=8, display=True):
    xp = Backend.get_xp_module()

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
        images = (image.copy() for _ in range(n))

        # modify each image:
        psf = nikon16x08na()
        psf = psf.astype(dtype=image.dtype, copy=False)
        images = (fft_convolve(image, psf) for image in images)
        images = (image + random.uniform(-10, 10) for image in images)

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

            dataset.set_slicing((slice(2, 3), ...))

            source_array = dataset.get_array("channel")

        with asection("Deconvolve..."):
            # output_folder:
            output_path = tmpdir / "deconv.zarr"
            output_dataset = ZDataset(path=output_path, mode="w", parent=dataset)

            # Do deconvolution:
            dataset_deconv(
                input_dataset=dataset,
                output_dataset=output_dataset,
                channels=("channel",),
                tilesize=320,
                method="lr",
                num_iterations=None,
                max_correction=None,
                power=1.0,
                blind_spot=0,
                back_projection="tpsf",
                wb_order=5,
                psf_objective="nikon16x08na",
                psf_na=None,
                psf_dxy=0.485,
                psf_dz=4 * 0.485,
                psf_xy_size=31,
                psf_z_size=31,
                scaling=(1, 1, 1),
                devices=[0],
                psf_show=False,
            )

            output_dataset.close()

            deconv_array = output_dataset.get_array("channel")

            assert deconv_array.shape[0] == 1
            assert deconv_array.shape[1:] == source_array.shape[1:]

        if display:

            def _c(array):
                return Backend.to_numpy(array)

            import napari

            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(_c(source_array), name="source_array")
            viewer.add_image(_c(deconv_array), name="deconv_array")
            viewer.grid.enabled = True
            viewer.dims.set_point(0, 0)
            napari.run()


if __name__ == "__main__":
    if not demo_deconv_cupy():
        demo_deconv_numpy()
