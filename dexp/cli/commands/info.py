import click
from arbol.arbol import aprint, asection

from dexp.cli.utils import _get_dataset_from_path


@click.command()
@click.argument('input_path')
def info(input_path):
    input_dataset = _get_dataset_from_path(input_path)
    with asection(f"Information on dataset at: {input_path}"):
        aprint(input_dataset.info())
        input_dataset.close()
        aprint("Done!")
