import click

from dexp.cli.main import _get_dataset_from_path


@click.command()
@click.argument('input_path')
def info(input_path):
    input_dataset = _get_dataset_from_path(input_path)
    print(input_dataset.info())
    input_dataset.close()
    print("Done!")
