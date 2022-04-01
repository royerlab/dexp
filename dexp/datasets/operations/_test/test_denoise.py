from pathlib import Path

import pytest
from arbol import asection

from dexp.cli.parsing import parse_devices
from dexp.datasets import ZDataset
from dexp.datasets.operations.denoise import dataset_denoise
from dexp.utils.backends.cupy_backend import is_cupy_available


@pytest.mark.parametrize(
    "dexp_nuclei_background_data",
    [dict(add_noise=True)],
    indirect=True,
)
def test_dataset_denoise(dexp_nuclei_background_data, tmp_path: Path, display_test: bool):

    if not is_cupy_available():
        pytest.skip(f"Cupy not found. Skipping {test_dataset_denoise.__name__} gpu test.")

    # Load
    _, _, image = dexp_nuclei_background_data

    channel = "channel"
    n_time_pts = 2
    input_path = tmp_path / "in_ds.zarr"
    output_path = tmp_path / "out_ds.zarr"

    with asection("Creating temporary zdatasets ..."):
        in_ds = ZDataset(input_path, mode="w")
        in_ds.add_channel(name=channel, shape=(n_time_pts,) + image.shape, dtype=image.dtype)

        for t in range(n_time_pts):
            in_ds.write_stack(channel, t, image)

        out_ds = ZDataset(output_path, mode="w")

    with asection("Executing command `dexp denoise ...`"):
        dataset_denoise(in_ds, out_ds, channels=[channel], tilesize=320, devices=parse_devices("all"))

    if display_test:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(in_ds.get_array(channel), name="input")
        viewer.add_image(out_ds.get_array(channel), name="output")
        viewer.grid.enabled = True

        napari.run()

    in_ds.close()
    out_ds.close()


if __name__ == "__main__":
    # the same as executing from the CLI
    # pytest <file name> -s --display True
    pytest.main([__file__, "-s", "--display", "True"])
