from time import time

import click
import numpy
from napari import Viewer, gui_qt
from numpy import s_

from dexp.datasets.clearcontrol_dataset import CCDataset
from dexp.datasets.zarr_dataset import ZDataset

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group()
def cli():
    print("------------------------------------------")
    print("  DEXP -- Data EXploration & Processing   ")
    print("  Royer lab                               ")
    print("------------------------------------------")
    print("")

    pass


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')  #
@click.option('--slice', '-s', default=None , help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--codec', '-z', default='zstd', help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ')  #
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target')  # , help='dataset slice'
@click.option('--project', '-p', type=int, default=1, help='max projection over given axis (0->T, 1->Z, 2->Y, 3->X)')  # , help='dataset slice'
def copy(input_path, output_path, channels, slice, codec, overwrite, project):
    if input_path.endswith('.zarr'):
        input_dataset = ZDataset(input_path)
    else:
        input_dataset = CCDataset(input_path)

    print(f"Available Channels: {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = get_folder_name_without_end_slash(input_path) + '.zarr'

    if not slice is None:
        dummy = s_[1, 2]
        slice = eval(f"s_{slice}")

    print(f"Requested channels  {channels if channels else '--All--'} ")

    if not channels is None:
        channels = channels.split(',')

    print(f"Selected channel: '{channels}' and slice: {slice}")

    print("Converting dataset.")
    print(f"Saving dataset to: {output_path} with zarr format... ")
    time_start = time()
    input_dataset.copy(output_path,
                       channels=channels,
                       slice=slice,
                       compression=codec,
                       overwrite=overwrite,
                       project=project)
    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    print("Done!")

    pass


def get_folder_name_without_end_slash(input_path):
    if input_path.endswith('/') or input_path.endswith('\\'):
        input_path = input_path[:-1]
    return input_path


@click.command()
@click.argument('input_path')
def info(input_path):
    if input_path.endswith('.zarr'):
        input_dataset = ZDataset(input_path)
    else:
        input_dataset = CCDataset(input_path)
    print(input_dataset.info())
    pass


@click.command()
@click.argument('input_path')
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slice', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).')
def view(input_path, channels, slice):
    if input_path.endswith('.zarr'):
        input_dataset = ZDataset(input_path)
    else:
        input_dataset = CCDataset(input_path)

    if channels is None:
        selected_channels = input_dataset.channels()
    else:
        channels = channels.split(',')
        selected_channels = list(set(channels) & set(input_dataset.channels()))

    if not slice is None:
        # do not remove dummy, this is to ensure that import is there...
        dummy = s_[1, 2]
        slice = eval(f"s_{slice}")

    print(f"Available channels: {input_dataset.channels()}")
    print(f"Requested channels: {channels}")
    print(f"Selected channels:  {selected_channels}")

    # Annoying napari induced warnings:
    import warnings
    warnings.filterwarnings("ignore")

    with gui_qt():
        viewer = Viewer()

        for channel in input_dataset.channels():
            print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")
            print(input_dataset.info(channel))

            array = input_dataset.get_stacks(channel)

            if slice:
                array = array[slice]

            print(f"Adding array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

            first_stack = numpy.array(input_dataset.get_stack(channel, 0, per_z_slice=False))

            print(f"Computing min and max from first stack...")
            min_value = first_stack.min()
            max_value = first_stack.max()
            print(f"min={min_value} and max={max_value}.")

            viewer.add_image(array, name=channel, clim_range=[max(0, min_value - 100), max_value + 100])

    pass


cli.add_command(copy)
cli.add_command(info)
cli.add_command(view)
