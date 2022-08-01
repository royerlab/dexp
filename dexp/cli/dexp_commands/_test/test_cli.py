from typing import Sequence

import pytest

from dexp.cli.dexp_main import cli
from dexp.utils.testing import cupy_only


def _run_main(commands: Sequence[str]) -> None:
    try:
        cli(commands)
    except SystemExit as exit:
        assert exit.code == 0


@pytest.mark.parametrize(
    "dexp_zarr_path",
    [dict(dataset_type="nuclei", dtype="uint16")],
    indirect=True,
)
@pytest.mark.parametrize(
    "command",
    [
        ["info"],
        ["check"],
        ["background", "-c", "image", "-a", "binary_area_threshold=500"],
        ["copy", "-wk", "2", "-s", "[:,:,40:240,40:240]"],
        ["deconv", "-c", "image", "-m", "lr", "-bp", "wb", "-i", "3"],
        ["deconv", "-c", "image", "-m", "admm", "-w"],
        ["denoise", "-c", "image"],
        ["extract-psf", "-pt", "100", "-c", "image"],
        ["generic", "-f", "gaussian_filter", "-pkg", "cupyx.scipy.ndimage", "-ts", "100", "-a", "sigma=1,2,2"],
        ["histogram", "-c", "image", "-m", "0"],
        ["projrender", "-c", "image", "-lt", "test projrender", "-lsi", "0.5"],
        ["segment", "-dc", "image"],
        ["tiff", "--split", "-wk", "2"],
        ["tiff", "-p", "0", "-wk", "2"],
        ["view", "-q", "-d", "2"],
    ],
)
@cupy_only
def test_cli_commands_nuclei_dataset(command: Sequence[str], dexp_zarr_path: str) -> None:
    _run_main(command + [dexp_zarr_path])


@pytest.mark.parametrize(
    "dexp_zarr_path",
    [dict(dataset_type="nuclei", n_time_pts=10, dtype="uint16")],
    indirect=True,
)
@pytest.mark.parametrize("command", [["stabilize", "-mr", "2", "-wk", "2"]])
@cupy_only
def test_cli_commands_long_nuclei_dataset(command: Sequence[str], dexp_zarr_path: str) -> None:
    _run_main(command + [dexp_zarr_path])


@pytest.mark.parametrize(
    "dexp_zarr_path",
    [dict(dataset_type="fusion", dtype="uint16")],
    indirect=True,
)
@pytest.mark.parametrize(
    "command",
    [
        ["register", "-c", "image-C0L0,image-C1L0"],
        ["fuse", "-c", "image-C0L0,image-C1L0", "-lr"],  # must be after registration
    ],
)
@cupy_only
def test_cli_commands_fusion_dataset(command: Sequence[str], dexp_zarr_path: str) -> None:
    _run_main(command + [dexp_zarr_path])


@pytest.mark.parametrize(
    "dexp_zarr_path",
    [dict(dataset_type="nuclei-skewed", dtype="uint16")],
    indirect=True,
)
@pytest.mark.parametrize(
    "command",
    [
        ["deskew", "-c", "image", "-xx", "1", "-zz", "1", "-a", "45"],
    ],
)
@cupy_only
def test_cli_commands_skewed_dataset(command: Sequence[str], dexp_zarr_path: str) -> None:
    _run_main(command + [dexp_zarr_path])


@pytest.mark.parametrize(
    "dexp_zarr_path",
    [dict(dataset_type="fusion-skewed", dtype="uint16")],
    indirect=True,
)
@pytest.mark.parametrize(
    "command",
    [
        ["fuse", "-c", "image-C0L0,image-C1L0", "-m", "mvsols", "-zpa", "0, 0"],
    ],
)
@cupy_only
def test_cli_commands_fusion_skewed_dataset(command: Sequence[str], dexp_zarr_path: str) -> None:
    _run_main(command + [dexp_zarr_path])
