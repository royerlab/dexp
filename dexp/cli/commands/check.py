import click
from arbol.arbol import asection, aprint

from dexp.cli.utils import _get_dataset_from_path, _parse_channels


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
def check(input_path, channels):
    input_dataset = _get_dataset_from_path(input_path)
    channels = _parse_channels(input_dataset, channels)

    with asection(f"checking integrity of: {input_path}, channels: {channels}"):
        result = input_dataset.check_integrity(channels)
        if result:
            aprint(f"No problem detected.")
        else:
            aprint(f"!!! PROBLEM DETECTED, CORRUPTION LIKELY !!!")

        input_dataset.close()
        aprint("Done!")
