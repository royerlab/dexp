import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import _parse_channels, _parse_slicing
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.view import dataset_view
from dexp.datasets.operations.view_remote import dataset_view_remote


@click.command()
@click.argument('input_paths', nargs=-1)
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).')
@click.option('--aspect', '-a', type=float, default=4, help='sets aspect ratio e.g. 4', show_default=True)
@click.option('--rescale-time', '-rt', is_flag=True, help='Rescale time axis, useful when channels acquisition rate was not uniform.', show_default=True)
@click.option('--clim', '-cl', type=str, default='0,512', help='Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]', show_default=True)
@click.option('--colormap', '-cm', type=str, default='viridis', help='sets colormap, e.g. viridis, gray, magma, plasma, inferno ', show_default=True)
@click.option('--windowsize', '-ws', type=int, default=1536, help='Sets the napari window size. i.e. -ws 400 sets the window to 400x400', show_default=True)
@click.option('--projectionsonly', '-po', is_flag=True, help='To view only the projections, if present.', show_default=True)
@click.option('--volumeonly', '-vo', is_flag=True, help='To view only the volumetric data.', show_default=True)
def view(input_paths,
         channels,
         slicing,
         aspect,
         rescale_time,
         clim,
         colormap,
         windowsize,
         projectionsonly,
         volumeonly
         ):
    """ Views dataset using napari (napari.org)
    """

    slicing = _parse_slicing(slicing)

    name = input_paths[0] + '...' if len(input_paths) > 1 else ''

    contrast_limits = tuple(float(v.strip()) for v in clim.split(','))

    if len(input_paths) == 1 and 'http' in input_paths[0]:

        with asection(f"Viewing dataset at: {input_paths}, channels: {channels}, slicing: {slicing}, aspect:{aspect} "):
            dataset_view_remote(input_path=input_paths[0],
                         name=name,
                         aspect=aspect,
                         channels=channels,
                         contrast_limits=contrast_limits,
                         colormap=colormap,
                         slicing=slicing,
                         windowsize=windowsize)
            aprint("Done!")

    else:
        input_dataset, input_paths = glob_datasets(input_paths)
        channels = _parse_channels(input_dataset, channels)

        with asection(f"Viewing dataset at: {input_paths}, channels: {channels}, slicing: {slicing}, aspect:{aspect} "):
            dataset_view(input_dataset=input_dataset,
                         name=name,
                         aspect=aspect,
                         channels=channels,
                         contrast_limits=contrast_limits,
                         colormap=colormap,
                         slicing=slicing,
                         windowsize=windowsize,
                         projections_only=projectionsonly,
                         volume_only=volumeonly,
                         rescale_time=rescale_time)
            input_dataset.close()
            aprint("Done!")
