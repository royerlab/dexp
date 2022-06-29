import click
from arbol import asection

from dexp.cli.parsing import (
    input_dataset_argument,
    output_dataset_options,
    workers_option,
)
from dexp.datasets import BaseDataset, ZDataset
from dexp.datasets.operations.fromraw import dataset_fromraw


@click.command()
@input_dataset_argument()
@output_dataset_options()
@workers_option()
@click.option("--ch-prefix", "-c", default=None, help="Prefix of channels, usually according to wavelength", type=str)
def fromraw(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    ch_prefix: str,
    workers: int,
):
    """Copies a dataset, channels can be selected, cropping can be performed, compression can be changed, ..."""

    with asection(
        f"Copying `fromraw`: {input_dataset.path} to {output_dataset.path} for channels with prefix {ch_prefix}"
    ):
        dataset_fromraw(
            input_dataset,
            output_dataset,
            ch_prefix,
            workers=workers,
        )

        input_dataset.close()
