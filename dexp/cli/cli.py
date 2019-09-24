import os
from os.path import join, exists
from time import time

import click
import dask
import numpy
from napari import Viewer, gui_qt
from numpy import s_


from dexp.datasets.clearcontrol_dataset import CCDataset
from dexp.datasets.zarr_dataset import ZDataset

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])




def get_dataset_from_path(input_path):
    if exists(join(input_path, 'stacks')):
        input_dataset = CCDataset(input_path)
    else:
        input_dataset = ZDataset(input_path)
    return input_dataset

def get_folder_name_without_end_slash(input_path):
    if input_path.endswith('/') or input_path.endswith('\\'):
        input_path = input_path[:-1]
    return input_path




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
@click.option('--project', '-p', type=int, default=None, help='max projection over given axis (0->T, 1->Z, 2->Y, 3->X)')  # , help='dataset slice'
def copy(input_path, output_path, channels, slice, codec, overwrite, project):
    input_dataset = get_dataset_from_path(input_path)

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


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--slice', '-s', default=None , help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--codec', '-z', default='zstd', help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ')  #
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target')  # , help='dataset slice'
def fuse(input_path, output_path, slice, codec, overwrite):
    input_dataset = get_dataset_from_path(input_path)

    print(f"Available Channels: {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = get_folder_name_without_end_slash(input_path) + '.zarr'

    if not slice is None:
        dummy = s_[1, 2]
        slice = eval(f"s_{slice}")


    print("Fusing dataset.")
    print(f"Saving dataset to: {output_path} with zarr format... ")
    time_start = time()
    input_dataset.fuse(output_path,
                       slice=slice,
                       compression=codec,
                       overwrite=overwrite
                       )
    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    print("Done!")

    pass

@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channel', '-c', default=None, help='selected channel.')  #
@click.option('--slice', '-s', default=None , help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target')  # , help='dataset slice'
def tiff(input_path, output_path, channel, slice, overwrite):
    input_dataset = get_dataset_from_path(input_path)

    print(f"Available Channels: {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = get_folder_name_without_end_slash(input_path) + '.zarr'

    if not slice is None:
        dummy = s_[1, 2]
        slice = eval(f"s_{slice}")


    print(f"Saving dataset to TIFF file: {output_path}")
    time_start = time()
    input_dataset.tiff(output_path,
                       channel=channel,
                       slice=slice,
                       overwrite=overwrite
                       )
    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    print("Done!")

    pass



@click.command()
@click.argument('input_path')
def info(input_path):
    input_dataset = get_dataset_from_path(input_path)
    print(input_dataset.info())
    pass


@click.command()
@click.argument('input_path')
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slice', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).')
@click.option('--volume', '-v', is_flag=True, help='to view with volume rendering (3D ray casting)')
def view(input_path, channels=None, slice=None, volume=False):
    input_dataset = get_dataset_from_path(input_path)

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

    if volume:

        array = input_dataset.get_stacks(selected_channels[0])[0]

        #from spimagine import volshow
        #volshow(array)

        #import yt
        #ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")
        #yt.interactive_render(ds)

        from vispy import app, scene, io
        from vispy.color import get_colormaps, BaseColormap
        from vispy.visuals.transforms import STTransform

        # Prepare canvas
        canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
        canvas.measure_fps()

        # Set up a viewbox to display the image with interactive pan/zoom
        view = canvas.central_widget.add_view()

        # Set whether we are emulating a 3D texture
        emulate_texture = False

        # Create the volume visuals, only one is visible
        volume1 = scene.visuals.Volume(array,
                                       parent=view.scene,
                                       threshold=0.225,
                                       emulate_texture=emulate_texture)

        volume1.transform = scene.STTransform(translate=(64, 64, 0))


        # Create three cameras (Fly, Turntable and Arcball)
        fov = 60.

        view.camera = scene.cameras.ArcballCamera(parent=view.scene, fov=fov, name='Arcball')

        app.run()

    else:

        # Annoying napari induced warnings:
        import warnings
        warnings.filterwarnings("ignore")

        with gui_qt():
            viewer = Viewer()

            for channel in selected_channels:
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

                # flip x for second camera:
                if 'C1' in channel:
                    array = dask.array.flip(array,-1)

                if 'C0' in channel:
                    colormap = 'red'
                elif 'C1' in channel:
                    colormap = 'blue'
                else:
                    colormap = 'viridis'

                viewer.add_image(array,
                                 name=channel,
                                 contrast_limits=[max(0, min_value - 100), max_value + 100],
                                 blending='additive',
                                 colormap=colormap)


    pass



cli.add_command(copy)
cli.add_command(fuse)
cli.add_command(info)
cli.add_command(tiff)
cli.add_command(view)
