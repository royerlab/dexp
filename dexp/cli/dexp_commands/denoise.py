from typing import Sequence, Union

import click
from arbol import asection

from dexp.cli.parsing import (
    channels_option,
    devices_option,
    input_dataset_argument,
    output_dataset_options,
    slicing_option,
)
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.operations.denoise import dataset_denoise
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@input_dataset_argument()
@output_dataset_options()
@channels_option()
@devices_option()
@slicing_option()
@click.option("--tilesize", "-ts", type=int, default=320, help="Tile size for tiled computation", show_default=True)
def denoise(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    tilesize: int,
    devices: Union[str, Sequence[int]],
):
    """Denoises input image using butterworth filter, parameters are estimated automatically
    using noise2self j-invariant cross-validation loss.
    """
    with asection(f"Denoising data to {output_dataset.path} for channels {channels}"):
        dataset_denoise(
            input_dataset,
            output_dataset,
            channels=channels,
            tilesize=tilesize,
            devices=devices,
        )

    input_dataset.close()
    output_dataset.close()
