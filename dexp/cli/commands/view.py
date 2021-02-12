import click
from arbol.arbol import aprint, asection
from zarr.errors import ArrayNotFoundError

from dexp.cli.utils import _parse_channels, _parse_slicing
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.view import dataset_view


@click.command()
@click.argument('input_paths', nargs=-1)
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).')
@click.option('--aspect', '-a', type=float, default=4, help='sets aspect ratio e.g. 4', show_default=True)
@click.option('--colormap', '-cm', type=str, default='viridis', help='sets colormap, e.g. viridis, gray, magma, plasma, inferno ', show_default=True)
@click.option('--windowsize', '-ws', type=int, default=1536, help='Sets the napari window size. i.e. -ws 400 sets the window to 400x400', show_default=True)
@click.option('--clim', '-cl', type=str, default=None, help='Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]', show_default=True)
def view(input_paths,
         channels=None,
         slicing=None,
         aspect=None,
         colormap='viridis',
         windowsize=1536,
         clim=None):
    slicing = _parse_slicing(slicing)

    if len(input_paths) == 1 and 'http' in input_paths[0]:
        if channels is None:
            aprint("Channel(s) must be specified!")
            return

        input_path = input_paths[0]

        if ':' not in input_path.replace("http://", ""):
            input_path = f'{input_path}:8000'

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

                    if slicing is not None:
                        array = array[slicing]

                    viewer.add_image(array, name=channel, visible=True)

                    try:
                        for axis in range(array.ndim):
                            if '/' in channel:
                                proj_array = da.from_zarr(f"{input_path}/{channel}_max{axis}")
                            else:
                                proj_array = da.from_zarr(f"{input_path}/{channel}/{channel}_max{axis}")
                            viewer.add_image(proj_array, name=f'{channel}_max{axis}', visible=True)

                    except (KeyError, ArrayNotFoundError):
                        aprint("Warning: could not find projections in dataset!")


    else:
        input_dataset, input_paths = glob_datasets(input_paths)
        channels = _parse_channels(input_dataset, channels)

        name = input_paths[0] + '...' if len(input_paths) > 1 else ''

        with asection(f"Viewing dataset at: {input_paths}, channels: {channels}, slicing: {slicing}, aspect:{aspect} "):
            dataset_view(input_dataset,
                         name,
                         aspect,
                         channels,
                         clim,
                         colormap,
                         slicing,
                         windowsize)
            input_dataset.close()
            aprint("Done!")
