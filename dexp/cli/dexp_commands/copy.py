from typing import Sequence

import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import DEFAULT_WORKERS_BACKEND
from dexp.cli.parsing import (
    channels_option,
    input_dataset_argument,
    output_dataset_options,
    slicing_option,
    workers_option,
)
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.operations.copy import dataset_copy
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@input_dataset_argument()
@output_dataset_options()
@channels_option()
@slicing_option()
@workers_option()
@click.option(
    "--zerolevel",
    "-zl",
    type=int,
    default=0,
    help="‘zero-level’ i.e. the pixel values in the restoration (to be substracted)",
    show_default=True,
)
@click.option(
    "--workersbackend",
    "-wkb",
    type=str,
    default=DEFAULT_WORKERS_BACKEND,
    help="What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ",
    show_default=True,
)
def copy(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    zerolevel: int,
    workers: int,
    workersbackend: str,
):
    """Copies a dataset, channels can be selected, cropping can be performed, compression can be changed, ..."""

    with asection(
        f"Copying from: {input_dataset.path} to {output_dataset.path} for channels: {channels}, slicing: {input_dataset.slicing} "
    ):
        dataset_copy(
            input_dataset=input_dataset,
            output_dataset=output_dataset,
            channels=channels,
            zerolevel=zerolevel,
            workers=workers,
            workersbackend=workersbackend,
        )

        input_dataset.close()
        aprint("Done!")
