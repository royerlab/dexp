import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import _default_workers_backend
from dexp.cli.parsing import _parse_channels, _parse_slicing, _get_output_path, _parse_devices
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.projrender import dataset_projection_rendering


@click.command()
@click.argument('input_paths', nargs=-1)
@click.option('--output_path', '-o', type=str, default=None, help='Output folder to store rendered PNGs. Default is: frames_<channel_name>')
@click.option('--channels', '-c', type=str, default=None, help='list of channels to color, all channels when ommited.')
@click.option('--slicing', '-s', type=str, default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).')
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--axis', '-ax', type=int, default=0, help='Sets the projection axis: 0->Z, 1->Y, 2->X ', show_default=True)  # , help='dataset slice'
@click.option('--dir', '-di', type=int, default=-1, help='Sets the projection direction: -1 -> top to bottom, +1 -> bottom to top.', show_default=True)  # , help='dataset slice'
@click.option('--mode', '-m', type=str, default='colormax',
              help='Sets the projection mode: ‘max’: classic max projection, ‘colormax’: color max projection, i.e. color codes for depth, ‘maxcolor’ same as colormax but first does depth-coding by color and then max projects (acheives some level of transparency). ',
              show_default=True)
@click.option('--clim', '-cl', type=str, default=None, help='Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]')
@click.option('--attenuation', '-at', type=float, default=0.1, help='Sets the projection attenuation coefficient, should be within [0, 1] ideally close to 0. Larger values mean more attenuation.',
              show_default=True)  # , help='dataset slice'
@click.option('--gamma', '-g', type=float, default=1.0, help='Sets the gamma coefficient pre-applied to the raw voxel values (before projection or any subsequent processing).', show_default=True)  # , help='dataset slice'
@click.option('--dlim', '-dl', type=str, default=None,
              help='Sets the depth limits. Depth limits. For example, a value of (0.1, 0.7) means that the colormap start at a normalised depth of 0.1, and ends at a normalised depth of 0.7, other values are clipped. Only used for colormax mode.',
              show_default=True)  # , help='dataset slice'
@click.option('--colormap', '-cm', type=str, default=None, help='sets colormap, e.g. viridis, gray, magma, plasma, inferno. Use a rainbow colormap such as turbo, bmy, or rainbow (recommended) for color-coded depth modes. ',
              show_default=True)
@click.option('--rgbgamma', '-cg', type=float, default=1.0, help='Gamma correction applied to the resulting RGB image. Usefull to brighten image', show_default=True)
@click.option('--transparency', '-t', is_flag=True, help='Enables transparency output when possible. Good for rendering on white (e.g. on paper).', show_default=True)
@click.option('--legendsize', '-lsi', type=float, default=1.0, help='Multiplicative factor to control size of legend. If 0, no legend is generated.', show_default=True)
@click.option('--legendscale', '-lsc', type=float, default=1.0, help='Float that gives the scale in some unit of each voxel (along the projection direction). Only in color projection modes.', show_default=True)
@click.option('--legendtitle', '-lt', type=str, default='color-coded depth (voxels)', help='Title for the color-coded depth legend.', show_default=True)
@click.option('--legendtitlecolor', '-ltc', type=str, default='1,1,1,1', help='Legend title color as a tuple of normalised floats: R, G, B, A  (values between 0 and 1).', show_default=True)
@click.option('--legendposition', '-lp', type=str, default='bottom_left', help='Position of the legend in pixels in natural order: x,y. Can also be a string: bottom_left, bottom_right, top_left, or top_right.', show_default=True)
@click.option('--legendalpha', '-la', type=float, default=1, help='Transparency for legend (1 means opaque, 0 means completely transparent)', show_default=True)
@click.option('--step', '-sp', type=int, default=1, help='Process every ‘step’ frames.', show_default=True)
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
@click.option('--devices', '-d', type=str, default='0', help='Sets the CUDA devices id, e.g. 0,1,2 or ‘all’', show_default=True)  #
def projrender(input_paths,
               output_path,
               channels,
               slicing,
               overwrite,
               axis,
               dir,
               mode,
               clim,
               attenuation,
               gamma,
               dlim,
               colormap,
               rgbgamma,
               transparency,
               legendsize,
               legendscale,
               legendtitle,
               legendtitlecolor,
               legendposition,
               legendalpha,
               step,
               workers,
               workersbackend,
               devices,
               stop_at_exception=True
               ):
    """ Renders datatset using 2D projections.
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    channels = _parse_channels(input_dataset, channels)
    slicing = _parse_slicing(slicing)
    devices = _parse_devices(devices)

    output_path = _get_output_path(input_paths[0], output_path, f"_{mode}_projection")

    dlim = None if dlim is None else tuple(float(strvalue) for strvalue in dlim.split(','))
    legendtitlecolor = tuple(float(v) for v in legendtitlecolor.split(','))

    if ',' in legendposition:
        legendposition = tuple(float(strvalue) for strvalue in legendposition.split(','))

    with asection(f"Projection rendering of: {input_paths} to {output_path} for channels: {channels}, slicing: {slicing} "):
        dataset_projection_rendering(input_dataset,
                                     output_path,
                                     channels,
                                     slicing,
                                     overwrite,
                                     axis,
                                     dir,
                                     mode,
                                     clim,
                                     attenuation,
                                     gamma,
                                     dlim,
                                     colormap,
                                     rgbgamma,
                                     transparency,
                                     legendsize,
                                     legendscale,
                                     legendtitle,
                                     legendtitlecolor,
                                     legendposition,
                                     legendalpha,
                                     step,
                                     workers,
                                     workersbackend,
                                     devices,
                                     stop_at_exception)

        input_dataset.close()
        aprint("Done!")
