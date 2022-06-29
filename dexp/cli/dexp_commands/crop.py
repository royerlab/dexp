from typing import Sequence

import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import (
    channels_callback,
    channels_option,
    input_dataset_argument,
    output_dataset_options,
    workers_option,
)
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.operations.crop import dataset_crop
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@input_dataset_argument()
@output_dataset_options()
@channels_option()
@workers_option()
@click.option(
    "--quantile",
    "-q",
    default=0.99,
    type=click.FloatRange(0, 1),
    help="Quantile parameter for lower bound of brightness for thresholding.",
    show_default=True,
)
@click.option(
    "--reference-channel",
    "-rc",
    default=None,
    help="Reference channel to estimate cropping. If no provided it picks the first one.",
    callback=channels_callback,
)
def crop(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    quantile: float,
    reference_channel: str,
    workers: int,
):
    """Automatically crops the dataset given a reference channel."""
    reference_channel = reference_channel[0]

    with asection(
        f"Cropping from: {input_dataset.path} to {output_dataset.path} for channels: {channels}, "
        f"using channel {reference_channel} as a reference."
    ):
        dataset_crop(
            input_dataset,
            output_dataset,
            channels=channels,
            reference_channel=reference_channel,
            quantile=quantile,
            workers=workers,
        )

        input_dataset.close()
        aprint("Done!")
