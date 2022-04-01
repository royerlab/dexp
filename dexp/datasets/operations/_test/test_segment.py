import random
from pathlib import Path

import pytest
from arbol import asection

from dexp.cli.parsing import _parse_devices
from dexp.datasets import ZDataset
from dexp.datasets.operations.segment import dataset_segment
from dexp.utils.backends.backend import Backend
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

    n_time_pts = 3
    channels = [f"channels_{i}" for i in range(n_time_pts)]
    out_channel = "Segments"
    input_path = tmp_path / "in_ds.zarr"
    output_path = tmp_path / "out_ds.zarr"
    df_path = tmp_path / output_path.name.replace(".zarr", ".csv")

    xp = Backend.get_xp_module(image)
    images = [xp.zeros_like(image) for _ in range(n_time_pts)]

    # adding some jitter to the different channels to emulate chromatic aberration
    for im in images:
        jitter = [random.randint(0, 5) for _ in range(image.ndim)]
        src_slicing = tuple(slice(0, d - j) for j, d in zip(jitter, image.shape))
        dst_slicing = tuple(slice(j, None) for j in jitter)
        im[src_slicing] = image[dst_slicing]

    with asection("Creating temporary zdatasets ..."):
        in_ds = ZDataset(input_path, mode="w")

        for im, ch in zip(images, channels):
            in_ds.add_channel(name=ch, shape=(n_time_pts,) + image.shape, dtype=image.dtype)
            for t in range(n_time_pts):
                in_ds.write_stack(ch, t, im)

        out_ds = ZDataset(output_path, mode="w")

    z_scale = 4

    with asection("Executing command `dexp segment ...`"):
        dataset_segment(
            in_ds,
            out_ds,
            detection_channels=channels,
            features_channels=channels,
            out_channel=out_channel,
            devices=_parse_devices("all"),
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
        colors = ("red", "green", "cyan")
        for color, ch in zip(colors, channels):
            viewer.add_image(in_ds.get_array(ch), name=ch, scale=(z_scale, 1, 1), blending="additive", colormap=color)

        viewer.add_labels(out_ds.get_array(out_channel), name="output", scale=(z_scale, 1, 1))

        napari.run()

    in_ds.close()
    out_ds.close()


if __name__ == "__main__":
    from dexp.utils.testing import test_as_demo

    # the same as executing from the CLI
    # pytest <file name> -s --display True
    test_as_demo(__file__)
