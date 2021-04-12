import click
from arbol.arbol import asection, aprint

from dexp.cli.parsing import _parse_channels
from dexp.datasets.open_dataset import glob_datasets


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
def check(input_paths, channels):
    """ Checks the integrity of a dataset.
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    channels = _parse_channels(input_dataset, channels)

    with asection(f"checking integrity of datasets {input_paths}, channels: {channels}"):
        result = input_dataset.check_integrity(channels)
        input_dataset.close()

        if not result:
            aprint(f"!!! PROBLEM DETECTED, CORRUPTION LIKELY !!!")
            return

    aprint(f"No problem detected.")
    aprint("Done!")
