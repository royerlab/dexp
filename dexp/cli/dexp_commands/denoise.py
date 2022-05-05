from typing import Sequence, Union

import click
from arbol import asection

from dexp.cli.parsing import (
    channels_option,
    input_dataset_argument,
    multi_devices_option,
    output_dataset_options,
    slicing_option,
    tilesize_option,
)
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.operations.denoise import dataset_denoise
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@input_dataset_argument()
@output_dataset_options()
@channels_option()
@multi_devices_option()
@slicing_option()
@tilesize_option()
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
