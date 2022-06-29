from pathlib import Path
from typing import Optional, Sequence

import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import DEFAULT_WORKERS_BACKEND
from dexp.cli.parsing import (
    _get_output_path,
    channels_option,
    input_dataset_argument,
    slicing_option,
    workers_option,
)
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.operations.tiff import dataset_tiff


@click.command()
@input_dataset_argument()
@channels_option()
@slicing_option()
@workers_option()
@click.option("--output-path", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", "-w", is_flag=True, help="to force overwrite of target", show_default=True)
@click.option(
    "--project", "-p", type=int, default=None, help="max projection over given an spatial axis (0->Z, 1->Y, 2->X)"
)
@click.option(
    "--split",
    is_flag=True,
    help="Splits dataset along first dimension, be carefull, if you slice to a single time point this will split along z!",
)
@click.option(
    "--clevel", "-l", type=int, default=0, help="Compression level, 0 means no compression, max is 9", show_default=True
)
@click.option(
    "--workersbackend",
    "-wkb",
    type=str,
    default=DEFAULT_WORKERS_BACKEND,
    help="What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ",
    show_default=True,
)
def tiff(
    input_dataset: BaseDataset,
    output_path: Optional[Path],
    channels: Sequence[str],
    overwrite: bool,
    project: bool,
    split: bool,
    clevel: int,
    workers: int,
    workersbackend: str,
):
    """Exports dataset as TIFF file(s)."""

    if output_path is None:
        output_path = Path(_get_output_path(input_dataset.path, None))

    with asection(
        f"Exporting to TIFF datset: {input_dataset.path}, channels: {channels}, project:{project}, split:{split}"
    ):
        dataset_tiff(
            input_dataset,
            output_path,
            channels=channels,
            overwrite=overwrite,
            project=project,
            one_file_per_first_dim=split,
            clevel=clevel,
            workers=workers,
            workersbackend=workersbackend,
        )

        input_dataset.close()
        aprint("Done!")
