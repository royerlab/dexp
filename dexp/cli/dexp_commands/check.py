from typing import Sequence

import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import channels_option, input_dataset_argument
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@input_dataset_argument()
@channels_option()
def check(input_dataset: ZDataset, channels: Sequence[int]) -> None:
    """Checks the integrity of a dataset."""

    with asection(f"checking integrity of datasets {input_dataset.path}, channels: {channels}"):
        result = input_dataset.check_integrity(channels)
        input_dataset.close()

        if not result:
            aprint("!!! PROBLEM DETECTED, CORRUPTION LIKELY !!!")
            return

    aprint("No problem detected.")
    aprint("Done!")
