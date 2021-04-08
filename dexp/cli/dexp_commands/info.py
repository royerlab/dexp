import click
from arbol.arbol import aprint, asection

from dexp.datasets.open_dataset import glob_datasets


@click.command()
@click.argument('input_paths', nargs=-1)
def info(input_paths):
    """ Prints out extensive information about dataset.
    """

    input_dataset, input_paths = glob_datasets(input_paths)

    with asection(f"Information on dataset at: {input_paths}"):
        aprint(input_dataset.info())
        input_dataset.close()
        aprint("Done!")
