import click

from dexp.cli.utils import _get_dataset_from_path, _parse_channels
from dexp.utils.timeit import timeit


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
def check(input_path, channels):
    input_dataset = _get_dataset_from_path(input_path)
    channels = _parse_channels(input_dataset, channels)

    with timeit("checking integrity"):
        result = input_dataset.check_integrity(channels)
        if not result:
            print(f"!!! PROBLEM DETECTED, CORRUPTION LIKELY !!!")

    input_dataset.close()
    print("Done!")
