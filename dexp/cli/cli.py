from time import time

import click
from napari import Viewer
from napari.util import app_context
from numpy import s_

from dexp.datasets.clearcontrol_dataset import CCDataset
from dexp.datasets.zarr_dataset import ZDataset

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group()
def cli():
    pass


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None)  # , help='list of channels'
@click.option('--slice', '-s', default=None)  # , help='dataset slice'
@click.option('--overwrite', '-w', is_flag=True)  # , help='dataset slice'
def convert(input_path, output_path, channels, slice, overwrite):
    if input_path.endswith('.zarr'):
        print('You provided a dataset in the zarr format. No need to convert anything...')
        return

    input_dataset = CCDataset(input_path)

    print(f"Channels: {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = input_path[:-1] + '.zarr'

    if not slice is None:
        dummy = s_[1, 2]
        slice = eval(f"s_{slice}")

    print(f"Channels requested: '{channels}' ")

    if not channels is None:
        channels = channels.split(',')

    print(f"Selecting channel: '{channels}' and slice: {slice}")

    print("Converting dataset.")
    print(f"Saving dataset to: {output_path} with zarr format... ")
    time_start = time()
    input_dataset.to_zarr(output_path, channels=channels, slice=slice, overwrite=overwrite)
    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    print("Done!")

    pass


@click.command()
@click.argument('input_path')
def info(input_path):
    if not input_path.endswith('.zarr'):
        print('Not a Zarr file!')
        return

    input_dataset = ZDataset(input_path)

    print(f"Channels: {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")
        print(input_dataset.info(channel))

    pass


@click.command()
@click.argument('input_path')
@click.option('--channels', '-c', default=None)  # , help='list of channels'
def view(input_path, channels):
    if not input_path.endswith('.zarr'):
        print('Not a Zarr file!')
        return

    print(f"Channels requested: '{channels}' ")

    input_dataset = ZDataset(input_path)

    if channels is None:
        selected_channels = input_dataset.channels()
    else:
        channels = channels.split(',')
        selected_channels = list(set(channels) & set(input_dataset.channels()))

    print(f"Available channels: {input_dataset.channels()}")
    print(f"Requested channels: {channels}")
    print(f"Selected channels:  {selected_channels}")

    with app_context():
        viewer = Viewer()

        for channel in input_dataset.channels():
            print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")
            print(input_dataset.info(channel))

            array = input_dataset.get_stacks(channel)

            print(f"Adding array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

            first_stack = input_dataset.get_stack(channel, 0)

            min_value = 0  # first_stack.min()
            max_value = 1000  # first_stack.max()
            # , clim_range = [min_value, max_value]
            #viewer.add_pyramid([first_stack], name='image', clim_range=[0, 1000])
            viewer.add_image(array, name='channel', clim_range=[0, 1000])

    pass


cli.add_command(convert)
cli.add_command(info)
cli.add_command(view)
