from os.path import exists, join
from typing import Tuple

import click
from PIL import Image
from arbol.arbol import section, aprint, asection

from dexp.cli.utils import _parse_channels, _get_dataset_from_path, _parse_slicing
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.render.projection import rgb_project


@click.command()
@click.argument('input_path')
@click.option('--output_path', '-o', default=None, help='Output folder to store rendered PNGs. Default is: frames_<channel_name>')
@click.option('--channels', '-c', default=None, help='list of channels to render, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).')
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--axis', '-ax', type=int, default=0, help='Sets the projection axis: 0->Z, 1->Y, 2->X ', show_default=True)  # , help='dataset slice'
@click.option('--dir', '-d', type=int, default=-1, help='Sets the projection direction: -1 -> top to bottom, +1 -> bottom to top.', show_default=True)  # , help='dataset slice'
@click.option('--mode', '-m', type=str, default='color', help='Sets the projection mode: ‘max’ ', show_default=True)  # , help='dataset slice'
@click.option('--norm', '-nm', type=bool, default=True, help='Normalises each stack before rendering', show_default=True)
@click.option('--clim', '-cl', type=str, default=None, help='Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]')
@click.option('--attenuation', '-at', type=float, default=0.1, help='Sets the projection attenuation coefficient.', show_default=True)  # , help='dataset slice'
@click.option('--gamma', '-g', type=int, default=1, help='Sets the gamma coefficient.', show_default=True)  # , help='dataset slice'
@click.option('--colormap', '-cm', type=str, default='magma', help='sets colormap, e.g. viridis, gray, magma, plasma, inferno ', show_default=True)
@click.option('--rendersize', '-rs', type=Tuple[int, int], default=None, help='Sets the frame render size (height). Default: the frame size is unchanged =.', show_default=True)
@click.option('--step', '-sp', type=int, default=1, help='Process every ‘step’ frames.', show_default=True)
def projrender(input_path,
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
               colormap,
               rendersize,
               step):
    input_dataset = _get_dataset_from_path(input_path)
    channels = _parse_channels(input_dataset, channels)
    slicing = _parse_slicing(slicing)

    aprint(f"Projection rendering of: {input_path} to {output_path} for channels: {channels}, slicing: {slicing} ")

    for channel in channels:

        with asection(f"Channel '{channel}' shape: {input_dataset.shape(channel)}:"):
            aprint(input_dataset.info(channel))

        array = input_dataset.get_array(channel, wrap_with_dask=True)
        dtype = array.dtype

        if slicing:
            array = array[slicing]

        aprint(f"Rendering array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

        nbframes = array.shape[0]

        while section("Rendering:"):
            for i in range(0, nbframes, step):
                aprint(f"Frame     : {i}")

                filename = join(output_path, f"frame_{i:05}.png")

                if overwrite or not exists(filename):

                    aprint("Loading stack...")
                    stack = array[i].compute()

                    with NumpyBackend():
                        aprint("Normalising stack")
                        if clim is not None:
                            aprint(f"Using provided min and max for contrast limits: {clim}")
                            min_value, max_value = (float(strvalue) for strvalue in clim.split(','))
                            clim = (min_value, max_value)

                        projection = rgb_project(stack,
                                                 axis=axis,
                                                 dir=dir,
                                                 mode=mode,
                                                 attenuation=attenuation,
                                                 gamma=gamma,
                                                 clim=clim,
                                                 cmap=colormap)

                        image = Image.fromarray(projection)
                        image.save(filename)

                    aprint(f"Saving frame: {filename}")

    input_dataset.close()
    aprint("Done!")
