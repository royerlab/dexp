from pathlib import Path

import pytest
from arbol import asection

from dexp.cli.parsing import _parse_devices
from dexp.datasets import ZDataset
from dexp.datasets.operations.segment import dataset_segment
from dexp.utils.backends.cupy_backend import is_cupy_available


@pytest.mark.parametrize(
    "dexp_nuclei_background_data",
    [dict(add_noise=True, length_z_factor=4)],
    indirect=True,
)
def test_dataset_segment(dexp_nuclei_background_data, tmp_path: Path, display_test: bool):

    if not is_cupy_available():
        pytest.skip(f"Cupy not found. Skipping {test_dataset_segment.__name__} gpu test.")

    # Load
    _, _, image = dexp_nuclei_background_data

    channel = "channel"
    n_time_pts = 2
    input_path = tmp_path / "in_ds.zarr"
    output_path = tmp_path / "out_ds.zarr"
    df_path = tmp_path / output_path.name.replace(".zarr", ".csv")

    with asection("Creating temporary zdatasets ..."):
        in_ds = ZDataset(input_path, mode="w")
        in_ds.add_channel(name=channel, shape=(n_time_pts,) + image.shape, dtype=image.dtype)

        for t in range(n_time_pts):
            in_ds.write_stack(channel, t, image)

        out_ds = ZDataset(output_path, mode="w")

    suffix = "_labels"
    z_scale = 4

    with asection("Executing command `dexp denoise ...`"):
        dataset_segment(
            in_ds,
            out_ds,
            channels=[channel],
            devices=_parse_devices("all"),
            suffix=suffix,
            z_scale=z_scale,
            area_threshold=1e2,
            minimum_area=50,
            h_minima=1,
            compactness=0,
            use_edt=True,
        )

    assert df_path.exists()

    if display_test:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(in_ds.get_array(channel), name="input", scale=(z_scale, 1, 1))
        viewer.add_labels(out_ds.get_array(channel + suffix), name="output", scale=(z_scale, 1, 1))

        napari.run()

    in_ds.close()
    out_ds.close()


if __name__ == "__main__":
    # the same as executing from the CLI
    # pytest <file name> -s --display True
    pytest.main([__file__, "-s", "--display", "True"])
