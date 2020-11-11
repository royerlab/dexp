from time import time

import click

from dexp.cli.main import _get_dataset_from_path


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
def check(input_path, channels):
    input_dataset = _get_dataset_from_path(input_path)

    if channels is None:
        selected_channels = input_dataset.channels()
    else:
        channels = channels.split(',')
        selected_channels = list(set(channels) & set(input_dataset.channels()))

    print(f"Available channel(s)    : {input_dataset.channels()}")
    print(f"Requested channel(s)    : {channels}")
    print(f"Selected channel(s)     : {selected_channels}")

    time_start = time()
    result = input_dataset.check_integrity()
    time_stop = time()
    print(f"Elapsed time: {time_stop - time_start} seconds")
    if not result:
        print(f"!!! PROBLEM DETECTED, CORRUPTION LIKELY !!!")
    print("Done!")

    input_dataset.close()
