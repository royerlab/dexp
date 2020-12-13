import click
from arbol.arbol import aprint, asection

from dexp.cli.utils import _parse_channels, _get_dataset_from_path, _parse_slicing
from dexp.datasets.operations.view import dataset_view


@click.command()
@click.argument('input_path')
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).')
@click.option('--volume', '-v', is_flag=True, help='to view with volume rendering (3D ray casting)', show_default=True)
@click.option('--aspect', '-a', type=float, default=4, help='sets aspect ratio e.g. 4', show_default=True)
@click.option('--colormap', '-cm', type=str, default='viridis', help='sets colormap, e.g. viridis, gray, magma, plasma, inferno ', show_default=True)
@click.option('--windowsize', '-ws', type=int, default=1536, help='Sets the napari window size. i.e. -ws 400 sets the window to 400x400', show_default=True)
@click.option('--clim', '-cl', type=str, default=None, help='Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]', show_default=True)
def view(input_path, channels=None, slicing=None, volume=False, aspect=None, colormap='viridis', windowsize=1536, clim=None):

    if 'http' in input_path:
        if channels is None:
            aprint("Channel(s) must be specified!")
            return

        with asection(f"Viewing remote dataset at: {input_path}, channel(s): {channels}"):
            channels = tuple(channel.strip() for channel in channels.split(','))
            channels = list(set(channels))

            from napari import Viewer, gui_qt

            with gui_qt():
                viewer = Viewer()
                for channel in channels:
                    import dask.array as da
                    if '/' in channel:
                        array = da.from_zarr(f"{input_path}/{channel}")
                    else:
                        array = da.from_zarr(f"{input_path}/{channel}/{channel}")
                    viewer.add_image(array, name='channel', visible=True)
    else:

        input_dataset = _get_dataset_from_path(input_path)
        channels = _parse_channels(input_dataset, channels)
        slicing = _parse_slicing(slicing)

        with asection(f"Viewing dataset at: {input_path}, channels: {channels}, slicing: {slicing}, aspect:{aspect} "):

            dataset_view(aspect, channels, clim, colormap, input_dataset, input_path, slicing, volume, windowsize)
            input_dataset.close()
            aprint("Done!")


