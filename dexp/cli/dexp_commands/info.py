import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import input_dataset_argument
from dexp.datasets.base_dataset import BaseDataset


@click.command()
@input_dataset_argument()
def info(input_dataset: BaseDataset):
    """Prints out extensive information about dataset."""

    with asection(f"Information on dataset at: {input_dataset.path}"):
        aprint(input_dataset.info())
        input_dataset.close()
        aprint("Done!")
