from typing import Sequence, Tuple

import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import (
    channels_option,
    empty_channels_callback,
    input_dataset_argument,
)
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.joined_dataset import JoinedDataset
from dexp.datasets.operations.view import dataset_view


@click.command()
@input_dataset_argument()
@channels_option()
@click.option(
    "--downscale", "-d", type=int, default=1, help="Downscale value to speedup visualization", show_default=True
)
@click.option(
    "--clim",
    "-cl",
    type=str,
    default="0,512",
    help="Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]",
    show_default=True,
    callback=lambda x, y, clim: tuple(float(v.strip()) for v in clim.split(",")),
)
@click.option(
    "--colormap",
    "-cm",
    type=str,
    default="magma",
    help="sets colormap, e.g. viridis, gray, magma, plasma, inferno ",
    show_default=True,
)
@click.option(
    "--windowsize",
    "-ws",
    type=click.IntRange(min=1),
    default=1536,
    help="Sets the napari window size. i.e. -ws 400 sets the window to 400x400",
    show_default=True,
)
@click.option(
    "--projectionsonly", "-po", is_flag=True, help="To view only the projections, if present.", show_default=True
)
@click.option("--volumeonly", "-vo", is_flag=True, help="To view only the volumetric data.", show_default=True)
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode. Doesn't display GUI.")
@click.option(
    "--labels",
    "-l",
    type=str,
    default=None,
    help="Channels to be displayed as labels layer",
    callback=empty_channels_callback,
)
def view(
    input_dataset: BaseDataset,
    channels: Sequence[str],
    clim: Tuple[float],
    downscale: int,
    colormap: str,
    windowsize: int,
    projectionsonly: bool,
    volumeonly: bool,
    quiet: bool,
    labels: Sequence[str],
):
    """Views dataset using napari (napari.org)"""

    name = input_dataset.path
    if isinstance(input_dataset, JoinedDataset):
        name = name + " ..."

    with asection(f"Viewing dataset at: {input_dataset.path}, channels: {channels}, downscale: {downscale}"):
        dataset_view(
            input_dataset=input_dataset,
            name=name,
            channels=channels,
            contrast_limits=clim,
            colormap=colormap,
            scale=downscale,
            windowsize=windowsize,
            projections_only=projectionsonly,
            volume_only=volumeonly,
            quiet=quiet,
            labels=labels,
        )

        input_dataset.close()
        aprint("Done!")
