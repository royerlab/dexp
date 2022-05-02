import subprocess
from typing import Sequence

import pytest

from dexp.utils.testing import cupy_only


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
        ["copy", "-wk", "2", "-s", "[:,:,40:240,40:240]"],
        ["denoise", "-c", "image"],
        ["generic", "-f", "gaussian_filter", "-pkg", "cupyx.scipy.ndimage", "-ts", "100", "-a", "sigma=1,2,2"],
        ["histogram", "-c", "image", "-m", "0"],
        ["tiff", "--split", "-wk", "2"],
        ["tiff", "-p", "0", "-wk", "2"],
        ["segment", "-dc", "image"],
        ["view", "-q", "-d", "2"],
    ],
)
@cupy_only
def test_cli_commands_nuclei_dataset(command: Sequence[str], dexp_zarr_path: str) -> None:
    assert subprocess.run(["dexp"] + command + [dexp_zarr_path]).returncode == 0


@pytest.mark.parametrize(
    "dexp_zarr_path",
    [dict(dataset_type="nuclei", n_time_pts=10, dtype="uint16")],
    indirect=True,
)
@pytest.mark.parametrize("command", [["stabilize", "-mr", "2", "-wk", "2"]])
@cupy_only
def test_cli_commands_long_nuclei_dataset(command: Sequence[str], dexp_zarr_path: str) -> None:
    assert subprocess.run(["dexp"] + command + [dexp_zarr_path]).returncode == 0


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
    assert subprocess.run(["dexp"] + command + [dexp_zarr_path]).returncode == 0
