from typing import Sequence

import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import DEFAULT_STORE
from dexp.cli.parsing import (
    channels_option,
    input_dataset_argument,
    optional_channels_callback,
)
from dexp.datasets import ZDataset


@click.command()
@input_dataset_argument()
@channels_option()
@click.option("--output-path", "-o", required=True)
@click.option(
    "--rename-channels",
    "-rc",
    default=None,
    help="You can rename channels: e.g. if channels are ‘channel1,anotherc’ then ‘gfp,rfp’ would rename the ‘channel1’ channel to ‘gfp’, and ‘anotherc’ to ‘rfp’ ",
    callback=optional_channels_callback,
)
@click.option("--store", "-st", default=DEFAULT_STORE, help="Zarr store: ‘dir’, ‘ndir’, or ‘zip’", show_default=True)
@click.option("--overwrite", "-w", is_flag=True, help="Forces overwrite of target", show_default=True)
@click.option(
    "--projection", "-p/-np", is_flag=True, default=True, help="If flags should be copied.", show_default=True
)
def add(
    input_dataset: ZDataset,
    output_path: str,
    channels: Sequence[str],
    rename_channels: Sequence[str],
    store: str,
    overwrite: bool,
    projection: bool,
) -> None:
    """Adds the channels selected from INPUT_PATHS to the given output dataset (created if not existing)."""

    with asection(
        f"Adding channels: {channels} from: {input_dataset.path} to {output_path}, with new names: {rename_channels}"
    ):
        input_dataset.add_channels_to(
            output_path,
            channels=channels,
            rename=rename_channels,
            store=store,
            overwrite=overwrite,
            add_projections=projection,
        )

        input_dataset.close()
        aprint("Done!")
