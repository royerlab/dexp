import math
import os
from os import listdir
from os.path import join, exists, isfile
from time import time

import click
import dask
import imageio
import numpy
from numpy import s_
from scipy import misc, special

from dexp.datasets.clearcontrol_dataset import CCDataset
from dexp.datasets.zarr_dataset import ZDataset
from dexp.utils.timeit import timeit

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

_default_store = 'dir'
_default_clevel = 3
_default_codec = 'zstd'


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
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None , help='Dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--store', '-st', default=_default_store, help='Store: ‘dir’, ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='Compression codec: zstd for ’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
@click.option('--project', '-p', type=int, default=None, help='max projection over given axis (0->T, 1->Z, 2->Y, 3->X)')
@click.option('--workers', '-k', default=1, help='Number of worker threads to spawn.', show_default=True)  #
def copy(input_path, output_path, channels, slicing, store, codec, clevel, overwrite, project, workers):
    input_dataset = get_dataset_from_path(input_path)

    print(f"Available Channel(s): {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = get_folder_name_without_end_slash(input_path) + '.zarr'

    if not slicing is None:
        print(f"Slicing: {slicing}")
        dummy = s_[1, 2]
        slicing = eval(f"s_{slicing}")

    print(f"Requested channel(s)  {channels if channels else '--All--'} ")
    if not channels is None:
        channels = channels.split(',')
    print(f"Selected channel(s): '{channels}' and slice: {slicing}")

    print("Converting dataset.")
    print(f"Saving dataset to: {output_path} with zarr format... ")
    time_start = time()
    input_dataset.copy(output_path,
                       channels=channels,
                       slicing=slicing,
                       store=store,
                       compression=codec,
                       compression_level=clevel,
                       overwrite=overwrite,
                       project=project,
                       workers=workers)
    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    print("Done!")



@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--rename', '-rc', default=None, help='You can rename channels: e.g. if channels are `channel1,anotherc` then `gfp,rfp` would rename the `channel1` channel to `gfp`, and `anotherc` to `rfp` ')
@click.option('--store', '-st', default=_default_store, help='Store: ‘dir’, ‘zip’', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
def add(input_path, output_path, channels, rename, store, overwrite):
    input_dataset = get_dataset_from_path(input_path)

    if channels is None:
        selected_channels = input_dataset.channels()
    else:
        channels = channels.split(',')
        selected_channels = list(set(channels) & set(input_dataset.channels()))

    if rename is None:
        rename = input_dataset.channels()
    else:
        rename = rename.split(',')

    print(f"Available channel(s)    : {input_dataset.channels()}")
    print(f"Requested channel(s)    : {channels}")
    print(f"Selected channel(s)     : {selected_channels}")
    print(f"New names for channel(s): {rename}")

    time_start = time()
    input_dataset.add(output_path,
                       channels=selected_channels,
                       rename=rename,
                       store=store,
                       overwrite=overwrite,
                       )
    time_stop = time()
    print(f"Elapsed time: {time_stop - time_start} seconds")
    print("Done!")

    pass






@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--slicing', '-s', default=None , help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--store', '-st', default=_default_store, help='Store: ‘dir’, ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ')
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--workers', '-k', type=int, default=1, help='Number of worker threads to spawn, recommended: 1 (unless you know what you are doing)', show_default=True)  #
@click.option('--zerolevel', '-zl', type=int, default=110,  help='Sets the \'zero-level\' i.e. the pixel values in the background (to be substracted)', show_default=True)  #
@click.option('--loadshifts', '-ls', is_flag=True, help='Turn on to load the registration parameters (i.e translation shifts) from another run', show_default=True)  #
def fuse(input_path, output_path, slicing, store, codec, clevel, overwrite, workers, zerolevel, loadshifts):
    input_dataset = get_dataset_from_path(input_path)

    print(f"Available Channels: {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = get_folder_name_without_end_slash(input_path) + '.zarr'

    if not slicing is None:
        print(f"Slicing: {slicing}")
        dummy = s_[1, 2]
        slicing = eval(f"s_{slicing}")

    print("Fusing dataset.")
    print(f"Saving dataset to: {output_path} with zarr format... ")
    time_start = time()
    input_dataset.fuse(output_path,
                       slicing=slicing,
                       store=store,
                       compression=codec,
                       compression_level=clevel,
                       overwrite=overwrite,
                       workers=workers,
                       zero_level=zerolevel,
                       load_shifts=loadshifts
                       )
    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    print("Done!")

    pass

@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None , help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--store', '-st', default=_default_store, help='Store: ‘dir’, ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)
@click.option('--workers', '-k', default=1, help='Number of worker threads to spawn, recommended: 1 (unless you know what you are doing)', show_default=True)
@click.option('--method', '-m', type=str, default='lr', help='Deconvolution method: for now only lr (Lucy Richardson)', show_default=True)
@click.option('--iterations', '-i', type=int, default=15, help='Number of deconvolution iterations. More iterations takes longer, will be sharper, but might also be potentially more noisy depending on method.', show_default=True)
@click.option('--maxcorrection', '-mc', type=int, default=2, help='Max correction in folds per iteration. Noisy datasets benefit from mc=2 (recommended), for noiseless datasets you can push to mc=8 or even more.', show_default=True)
@click.option('--power', '-pw', type=float, default=1.5, help='Correction exponent, default for standard LR is 1, set to 1.5 for acceleration.', show_default=True)
@click.option('--dxy', '-dxy', type=float, default=0.485, help='Voxel size along x and y in microns', show_default=True)
@click.option('--dz', '-dz', type=float, default=4*0.485, help='Voxel size along z in microns', show_default=True)
@click.option('--xysize', '-sxy', type=int, default=17, help='Voxel size along xy in microns', show_default=True)
@click.option('--zsize', '-sz', type=int, default=31, help='Voxel size along z in microns', show_default=True)
@click.option('--downscalexy2', '-d', is_flag=False, help='Downscales along x and y for faster deconvolution (but worse quality of course)', show_default=True)  #
def deconv(input_path, output_path, channels, slicing, store, codec, clevel, overwrite, workers, method, iterations, maxcorrection, power, dxy, dz, xysize, zsize, downscalexy2):
    input_dataset = get_dataset_from_path(input_path)

    print(f"Available Channels: {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = get_folder_name_without_end_slash(input_path) + '.zarr'


    if not slicing is None:
        print(f"Slicing: {slicing}")
        dummy = s_[1, 2]
        slicing = eval(f"s_{slicing}")

    print(f"Requested channel(s)  {channels if channels else '--All--'} ")
    if not channels is None:
        channels = channels.split(',')
    print(f"Selected channel(s): '{channels}' and slice: {slicing}")

    print("Fusing dataset.")
    print(f"Saving dataset to: {output_path} with zarr format... ")
    time_start = time()
    input_dataset.deconv(output_path,
                         channels=channels,
                         slicing=slicing,
                         store=store,
                         compression=codec,
                         compression_level=clevel,
                         overwrite=overwrite,
                         workers=workers,
                         method=method,
                         num_iterations=iterations,
                         max_correction=maxcorrection,
                         power=power,
                         dxy=dxy,
                         dz=dz,
                         xy_size=xysize,
                         z_size=zsize,
                         downscalexy2=downscalexy2
                         )

    # path,
    # channels=None,
    # slicing=None,
    # compression='zstd',
    # compression_level=3,
    # overwrite=False,
    # workers=None,
    # deconv_mode='lr',
    # num_iterations=20,
    # dxy=0.406,
    # dz=4*0.406,
    # psf_size=17,
    # downscalexy2=False
    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    print("Done!")

    pass


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--slicing', '-s', default=None , help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--store', '-st', default=_default_store, help='Store: ‘dir’, ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)  #
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--context', '-c', default='default', help="IsoNet context name", show_default=True)  # , help='dataset slice'
@click.option('--mode', '-m', default='pta', help="mode: 'pta' -> prepare, train and apply, 'a' just apply  ", show_default=True)  # , help='dataset slice'
@click.option('--max_epochs', '-e', type=int, default='50', help='to force overwrite of target', show_default=True)  # , help='dataset slice'
def isonet(input_path, output_path, slicing, store, codec, clevel, overwrite, context, mode, max_epochs):
    input_dataset = get_dataset_from_path(input_path)

    print(f"Available Channels: {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = get_folder_name_without_end_slash(input_path) + '.zarr'

    if not slicing is None:
        dummy = s_[1, 2]
        print(f"Slicing: {slicing}")
        slicing = eval(f"s_{slicing}")

    print("Fusing dataset.")
    print(f"Saving dataset to: {output_path} with zarr format... ")
    time_start = time()
    input_dataset.isonet(output_path,
                         slicing=slicing,
                         store=store,
                         compression=codec,
                         compression_level=clevel,
                         overwrite=overwrite,
                         context=context,
                         mode=mode,
                         max_epochs=max_epochs
                         )
    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    print("Done!")

    pass


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='selected channels.')  #
@click.option('--slicing', '-s', default=None , help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--project', '-p', type=int, default=None, help='max projection over given axis (0->T, 1->Z, 2->Y, 3->X)')  # , help='dataset slice'
@click.option('--split', is_flag=True, help='Splits dataset along first dimension, be carefull, if you slice to a single time point this will split along z!')  # , help='dataset slice'
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level, 0 means no compression, max is 9', show_default=True)  # , help='dataset slice'
@click.option('--workers', '-k', default=1, help='Number of worker threads to spawn.', show_default=True)  #
def tiff(input_path, output_path, channels, slicing, overwrite, project, split, compress, clevel, workers):
    input_dataset = get_dataset_from_path(input_path)

    print(f"Available Channel(s): {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = get_folder_name_without_end_slash(input_path) + '.zarr'

    if not slicing is None:
        dummy = s_[1, 2]
        print(f"Slicing: {slicing}")
        slicing = eval(f"s_{slicing}")


    print(f"Requested channel(s)  {channels if channels else '--All--'} ")

    if not channels is None:
        channels = channels.split(',')

    print(f"Selected channel(s): '{channels}' and slice: {slicing}")


    print(f"Saving dataset to TIFF file: {output_path}")
    time_start = time()
    input_dataset.tiff(output_path,
                       channels=channels,
                       slicing=slicing,
                       overwrite=overwrite,
                       project=project,
                       one_file_per_first_dim=split,
                       clevel=clevel,
                       workers=workers
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
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).')
@click.option('--volume', '-v', is_flag=True, help='to view with volume rendering (3D ray casting)', show_default=True)
@click.option('--aspect', '-a', type=float, default=4, help='sets aspect ratio e.g. 4', show_default=True)
@click.option('--colormap', '-cm', type=str, default='viridis', help='sets colormap, e.g. viridis, gray, magma, plasma, inferno ', show_default=True)
@click.option('--render', '-r', type=str, default=None, help='Renders video using napari movie script (not great, prefer the render command instead)')
@click.option('--windowsize', '-ws', type=int, default=1536, help='Sets the napari window size. i.e. -ws 400 sets the window to 400x400', show_default=True)
@click.option('--clim', '-cl', type=str, default=None, help='Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]', show_default=True)
def view(input_path, channels=None, slicing=None, volume=False, aspect=None, colormap='viridis', render=None, windowsize=1536, clim=None):
    
    from napari import Viewer, gui_qt
    from napari._qt.threading import thread_worker
        
    input_dataset = get_dataset_from_path(input_path)

    if channels is None:
        selected_channels = input_dataset.channels()
    else:
        channels = channels.split(',')
        selected_channels = list(set(channels) & set(input_dataset.channels()))

    if not slicing is None:
        # do not remove dummy, this is to ensure that import is there...
        dummy = s_[1, 2]
        print(f"Slicing: {slicing}")
        slicing = eval(f"s_{slicing}")

    print(f"Available channel(s): {input_dataset.channels()}")
    print(f"Requested channel(s): {channels}")
    print(f"Selected channel(s):  {selected_channels}")

    # Annoying napari induced warnings:
    import warnings
    warnings.filterwarnings("ignore")

    with gui_qt():
        viewer = Viewer(title=f"DEXP | viewing with napari: {input_path} ", ndisplay=3 if volume else 2)

        viewer.window.resize(windowsize+256, windowsize)

        for channel in selected_channels:
            print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")
            print(input_dataset.info(channel))

            array = input_dataset.get_stacks(channel)

            if slicing:
                array = array[slicing]

            print(f"Adding array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

            if clim is None:
                print(f"Computing min and max from first stack...")
                first_stack = numpy.array(input_dataset.get_stack(channel, 0, per_z_slice=False))[::8]
                min_value = numpy.percentile(first_stack, q=0.1)
                max_value = numpy.percentile(first_stack, q=99.99)
                print(f"min={min_value} and max={max_value}.")
                contrast_limits = [max(0, min_value - 32), max_value + 32]
            else:
                print(f"provided min and max for contrast limits: {clim}")
                min_value, max_value = ( float(strvalue) for strvalue in clim.split(','))
                contrast_limits = [min_value, max_value]


            # flip x for second camera:
            if 'C1' in channel:
                array = dask.array.flip(array,-1)

            layer = viewer.add_image(array,
                             name=channel,
                             contrast_limits=contrast_limits,
                             blending='additive',
                             colormap=colormap,
                             attenuation=0.04,
                             rendering='attenuated_mip')

            if not aspect is None:
                layer.scale[-3] = aspect
                print(f"Setting aspect ratio to {aspect} (layer.scale={layer.scale})")

            #For some reason spome parameters refuse to be set, this solves it:
            @thread_worker
            def workaround_for_recalcitrant_parameters():
                print("Setting 3D rendering parameters")
                layer.attenuation=0.02
                layer.rendering='attenuated_mip'

            worker = workaround_for_recalcitrant_parameters()
            worker.start()


            if render is not None:

                render = render.strip()
                parameters = dict(item.split("=") for item in render.split(",")) if render != 'defaults' else dict()

                backend = parameters['backend'] if 'backend' in parameters else 'naparimovie'

                if backend == 'naparimovie':
                    from naparimovie import Movie

                    script = parameters['script'] if 'script' in parameters else 'script.txt'
                    steps = int(parameters['steps']) if 'steps' in parameters else 60
                    res = int(parameters['res']) if 'res' in parameters else 1024
                    fps = int(parameters['fps']) if 'fps' in parameters else 60
                    name = parameters['name'] if 'name' in parameters else 'movie.mp4'

                    print(f"Movie Parameters provided: {parameters}")
                    print(f"Movie script: {script}, steps={steps}, res={res}, fps={fps}, name={name}")

                    #time.sleep(1)
                    movie = Movie(myviewer=viewer)
                    movie.inter_steps = steps
                    movie.create_state_dict_from_script(script)
                    movie.make_movie(name=name, resolution=res, fps=fps, show=False)



@click.command()
@click.argument('input_path')
@click.option('--output_path', '-o', default=None, help='Output folder to store rendered PNGs. Default is: frames_<channel_name>')
@click.option('--channels', '-c', default=None, help='list of channels to render, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z).')
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--aspect', '-a', type=float, default=4, help='sets aspect ratio e.g. 4', show_default=True)
@click.option('--colormap', '-cm', type=str, default='magma', help='sets colormap, e.g. viridis, gray, magma, plasma, inferno ', show_default=True)
@click.option('--rendersize', '-rs', type=int, default=2048, help='Sets the frame render size. i.e. -ws 400 sets the window to 400x400', show_default=True)
@click.option('--clim', '-cl', type=str, default='0,1024', help='Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]')
@click.option('--options', '-opt', type=str, default=None, help='Render options, e.g: \'gamma=1.2,box=True\'. Important: no white spaces!!! Complete list with defaults will be displayed on first run')
def render(input_path, output_path, channels=None, slicing=None, overwrite=False, aspect=None, colormap='viridis', rendersize=1536, clim=None, options=None):

    input_dataset = get_dataset_from_path(input_path)

    if channels is None:
        selected_channels = input_dataset.channels()
    else:
        channels = channels.split(',')
        selected_channels = list(set(channels) & set(input_dataset.channels()))

    if not slicing is None:
        # do not remove dummy, this is to ensure that import is there...
        dummy = s_[1, 2]
        print(f"Slicing: {slicing}")
        slicing = eval(f"s_{slicing}")

    print(f"Available channel(s): {input_dataset.channels()}")
    print(f"Requested channel(s): {channels}")
    print(f"Selected channel(s):  {selected_channels}")

    for channel in selected_channels:
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")
        print(input_dataset.info(channel))

        array = input_dataset.get_stacks(channel)
        dtype = array.dtype

        if slicing:
            array = array[slicing]

        print(f"Rendering array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

        if clim is None:
            print(f"Computing min and max from first stack...")
            first_stack = numpy.array(input_dataset.get_stack(channel, 0, per_z_slice=False))
            min_value = max(0, first_stack.min()- 100)
            max_value = first_stack.max()+100
            print(f"min={min_value} and max={max_value}.")
        else:
            print(f"provided min and max for contrast limits: {clim}")
            min_value, max_value = ( float(strvalue) for strvalue in clim.split(','))


        print(f"Provided rendering options: {options}")
        options = dict(item.split("=") for item in options.split(",")) if options is not None else dict()

        def str2bool(v):
            return v.lower() in ("yes", "true", "t", "1")

        nbtp = array.shape[0]
        nbframes = int(options['nbframes']) if 'nbframes' in options else 1010

        skip = int(options['skip']) if 'skip' in options else 1
        cut = str(options['cut']) if 'cut' in options else 'nocut'
        cutpos = float(options['cutpos']) if 'cutpos' in options else 0
        cutspeed = float(options['cutspeed']) if 'cutspeed' in options else 0
        irot  = options['irot'] if 'irot' in options else ''
        raxis  = options['raxis'] if 'raxis' in options else 'y'
        rstart = float(options['rstart']) if 'rstart' in options else 0
        rspeed = float(options['rspeed']) if 'rspeed' in options else 0.15
        tstart = int(options['tstart']) if 'tstart' in options else 0
        tspeed = float(options['tspeed']) if 'tspeed' in options else 0.5
        gamma = float(options['gamma']) if 'gamma' in options else 0.9
        zoom = float(options['zoom']) if 'zoom' in options else 1.45
        alpha = float(options['alpha']) if 'alpha' in options else 0.3
        box = str2bool(options['box']) if 'box' in options else False
        maxsteps = int(options['maxsteps']) if 'maxsteps' in options else 1024
        norm = str2bool(options['norm']) if 'norm' in options else True
        normrange = float(options['normrange']) if 'normrange' in options else 1024
        videofilename = options['name'] if 'name' in options else 'video.mp4'

        print(f"Video filename          : {videofilename}")
        print(f"Number of time points   : {nbtp}")
        print(f"Number of frames        : {nbframes}  \toption: nbframes: \tint")
        print(f"Render one frame every  : {skip}      \toption: skip:     \tint")
        print(f"Volume cutting          : {cut}       \toption: cut:      \t[left, right, top, bottom, front, back, none]")
        print(f"Volume cutting position : {cutpos}    \toption: cutpos:   \tfloat")
        print(f"Volume cutting speed    : {cutspeed}  \toption: cutspeed: \tfloat")
        print(f"Initial time point      : {tstart}    \toption: tstart:   \tint")
        print(f"Time    speed           : {tspeed}    \toption: tspeed:   \tfloat")
        print(f"Initial rotation        : {irot}      \toption: irot:     \t xxyzzz -> 90deg along x, 45deg along y, 135deg along z ")
        print(f"Rotation axis           : {raxis}     \toption: raxis:    \t[x,y,z]")
        print(f"Initial rotation angle  : {rstart}    \toption: rstart:   \tfloat")
        print(f"Rotation speed          : {rspeed}    \toption: rspeed:   \tfloat")
        print(f"Gamma                   : {gamma}     \toption: gamma:    \tfloat")
        print(f"Zoom                    : {zoom}      \toption: zoom:     \tfloat")
        print(f"Alpha                   : {alpha}     \toption: alpha:    \tfloat")
        print(f"box?                    : {box}       \toption: box:      \tbool (true/false)")
        print(f"normalisation           : {norm}      \toption: norm:     \tbool (true/false)")
        print(f"normalisation range     : {normrange} \toption: normrange:\tfloat")
        print(f"Max steps for vol render: {maxsteps}  \toption: maxsteps: \tint")

        if output_path is None:
            output_path = f"frames_{channel}"
        os.makedirs(output_path, exist_ok=True)
        from spimagine import volshow
        from spimagine import DataModel
        from spimagine import NumpyData

        print("Opening spimagine...")
        import spimagine
        spimagine.config.__DEFAULTMAXSTEPS__ = maxsteps
        spimagine.config.__DEFAULT_TEXTURE_WIDTH__ = rendersize

        datamodel = DataModel(NumpyData(array[0].compute()))
        print("Creating Spimagine window... (you can minimise but don't close!)")
        win = volshow(datamodel, stackUnits=(1., 1., aspect), autoscale=False, show_window=True)
        win.resize(rendersize+32, rendersize+32)
        win.showMinimized()


        for i in range(0, nbframes, skip):
            print(f"______________________________________________________________________________")
            print(f"Frame     : {i}")

            tp = tstart+int(tspeed * i)
            if tp >= nbtp:
                break
            print(f"Time point: {tp}")

            angle = rstart+rspeed*i
            print(f"Angle     : {angle}")

            effcutpos = cutpos + cutspeed

            filename = join(output_path, f"frame_{i:05}.png")

            if overwrite or not exists(filename):

                print("Loading stack...")
                stack = array[int(tp)].compute()

                if norm:
                    print("Computing percentile...")
                    rmax = numpy.percentile(stack[::8].astype(numpy.float32), q=99.99).astype(numpy.float32)
                    #rmax = numpy.max(stack[::8]).astype(numpy.float32)
                    print(f"rmax={rmax}")

                    print("Normalising...")
                    norm_max_value = normrange
                    norm_min_value = 64.0
                    #stack = norm_min_value+stack*((norm_max_value-norm_min_value)/rmax)
                    stack = stack*numpy.array((norm_max_value-norm_min_value)/rmax, dtype=numpy.float32)
                    stack += numpy.array(norm_min_value, dtype=dtype)
                    stack = stack.astype(dtype)

                #print("Opening spimagine...")
                #win = volshow(stack, stackUnits=(1., 1., aspect), autoscale=False, show_window=True)

                print("Loading stack into Spimagine...")
                datamodel.setContainer(NumpyData(stack))
                win.setModel(datamodel)

                print("Setting rendering parameters...")
                if colormap in spimagine.config.__COLORMAPDICT__:
                    win.set_colormap(colormap)
                else:
                    from matplotlib import colors
                    rgb = colors.to_rgba(colormap)[:3]
                    print(f"Turning the provided color: {colormap} = {rgb} into a colormap.")
                    win.set_colormap_rgb(rgb)

                win.transform.setStackUnits(1., 1., aspect)
                win.transform.setGamma(gamma)
                win.transform.setMin(min_value)
                win.transform.setMax(max_value)
                win.transform.setZoom(zoom)
                win.transform.setAlphaPow(alpha)
                win.transform.setBox(box)


                if cut=='left':
                    win.transform.setBounds(effcutpos, 1, -1, 1, -1, 1)
                elif cut=='right':
                    win.transform.setBounds(-1, effcutpos, -1, 1, -1, 1)
                elif cut=='top':
                    win.transform.setBounds(-1, 1, effcutpos, 1, -1, 1)
                elif cut=='bottom':
                    win.transform.setBounds(-1, 1, -1, effcutpos, -1, 1)
                elif cut=='front':
                    win.transform.setBounds(-1, 1, -1, 1, effcutpos, 1)
                elif cut=='back':
                    win.transform.setBounds(-1, 1, -1, 1, -1, effcutpos)
                elif cut=='centerx':
                    win.transform.setBounds(-0.25-effcutpos, 0.25+effcutpos, -1, 1, -1, 1)
                elif cut=='centery':
                    win.transform.setBounds(-1, 1, -0.25-effcutpos, 0.25+effcutpos, -1, 1)
                elif cut=='centerz':
                    win.transform.setBounds(-1, 1, -1, 1, -0.25-effcutpos, 0.25+effcutpos)
                elif cut=='none':
                    win.transform.setBounds(-1, 1, -1, 1, -1, 1)


                win.transform.setRotation(0, 1, 0, 0)

                for character in irot:
                    if character == 'x':
                        print(f"Rotating along x axis by 45 deg (prev quatRot={win.transform.quatRot})")
                        win.transform.addRotation(0.5* math.pi / 4, 1, 0, 0)
                    elif character == 'y':
                        print(f"Rotating along y axis by 45 deg (prev quatRot={win.transform.quatRot})")
                        win.transform.addRotation(0.5*math.pi / 4, 0, 1, 0)
                    elif character == 'z':
                        print(f"Rotating along z axis by 45 deg (prev quatRot={win.transform.quatRot})")
                        win.transform.addRotation(0.5*math.pi / 4, 0, 0, 1)
                print(f"Rotation after initial axis rotation: {win.transform.quatRot}")

                if 'x' in raxis:
                    print(f"Rotating along x axis by {angle} deg (prev quatRot={win.transform.quatRot})")
                    win.transform.addRotation(0.5*angle * (math.pi / 180), 1, 0, 0)
                if 'y' in raxis:
                    print(f"Rotating along y axis by {angle} deg (prev quatRot={win.transform.quatRot})")
                    win.transform.addRotation(0.5*angle * (math.pi / 180), 0, 1, 0)
                if 'z' in raxis:
                    print(f"Rotating along z axis by {angle} deg (prev quatRot={win.transform.quatRot})")
                    win.transform.addRotation(0.5*angle * (math.pi / 180), 0, 0, 1)

                print(f"Final rotation: {win.transform.quatRot}")




                print(f"Saving frame: {filename}")
                win.saveFrame(filename)

    win.closeMe()

    raise SystemExit
    import sys
    sys.exit()


@click.command()
@click.argument('input_path')
@click.option('--output_path', '-o', type=str, default=None, help='Output file path for movie')
@click.option('--framerate', '-fps', type=int, default=30, help='Sets the framerate in frames per second', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
def mp4(input_path, output_path, framerate, overwrite=False):

    if output_path is None:
        videofilepath = os.path.basename(os.path.normpath(input_path))+'.mp4'
    else:
        videofilepath = output_path

    videofilepath = videofilepath.replace('frames_','')

    if overwrite or not exists(videofilepath):

        ffmpeg_command = f"ffmpeg -framerate {framerate} -start_number 0 -pattern_type glob -i '{input_path}/*.png'  " \
                         f"-f mp4 -vcodec libx264 -preset slow -pix_fmt yuv420p -y {videofilepath}"
        #-vf  \"crop=576:1240:320:0\"

        os.system(ffmpeg_command)
    else:
        print(f"Video file: {videofilepath} already exists! use -w option to force overwrite...")


@click.command()
@click.argument('input_paths', nargs=-1)
@click.option('--output_path', '-o', type=str, default=None, help='Output folder for blended frames.')
@click.option('--blending', '-b', type=str, default='max', help='Blending mode: max, add, addclip, adderf (add stands for addclip).', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, set to -1 for maximum number of workers', show_default=True)  #
def blend(input_paths, output_path, blending, overwrite, workers):

        if workers <= 0:
            workers = os.cpu_count()//2

        if output_path is None:
            basename = '_'.join([os.path.basename(os.path.normpath(p)).replace('frames_','') for p in input_paths])
            output_path = 'frames_'+basename

        os.makedirs(output_path, exist_ok=True)

        if overwrite or not exists(output_path):

            first_folder = input_paths[0]
            pngfiles = [f for f in listdir(first_folder) if isfile(join(first_folder, f)) and f.endswith('.png')]
            pngfiles.sort()

            from joblib import Parallel, delayed

            def process(pngfile):
                with timeit('Elapsed time: '):
                    blended_image_array = None
                    original_dtype = None

                    for path in input_paths:

                        print(f"Reading file: {pngfile} from path: {path}")
                        try:
                            if isfile(path):
                                image_array = imageio.imread(path)
                            else:
                                image_array = imageio.imread(join(path, pngfile))
                        except FileNotFoundError:
                            print(f"WARNING: file {join(path, pngfile)} (or {path}) not found! using blank frame instead!")
                            image_array = numpy.zeros_like(blended_image_array)

                        if blended_image_array is None:
                            original_dtype = image_array.dtype
                            blended_image_array = image_array.astype(numpy.float32)

                        else:
                            if blending == 'max':
                                print(f"Blending using max mode.")
                                blended_image_array = numpy.maximum(blended_image_array, image_array)
                            elif blending == 'add' or  blending == 'addclip':
                                print(f"Blending using add mode (clipped).")
                                blended_image_array = numpy.clip(blended_image_array+image_array, 0, 255)
                            elif blending == 'adderf':
                                print(f"Blending using add mode (erf saturation).")
                                blended_image_array = 255*special.erf(blended_image_array+image_array/255)

                    print(f"Writing file: {pngfile} in folder: {output_path}")
                    imageio.imwrite(join(output_path, pngfile), blended_image_array.astype(original_dtype))

            Parallel(n_jobs=workers)(delayed(process)(pngfile) for pngfile in pngfiles)


        else:
            print(f"Folder: {output_path} already exists! use -w option to force overwrite...")

@click.command()
@click.argument('input_paths', nargs=-1)
@click.option('--output_path', '-o', type=str, default=None, help='Output folder for blended frames.')
@click.option('--orientation', '-r', type=str, default='horiz', help='Stitching mode: horiz, vert', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, set to -1 for maximum number of workers', show_default=True)  #
def stack(input_paths, output_path, orientation, overwrite, workers):

    if workers <=0 :
        workers = os.cpu_count()//2

    if output_path is None:
        basename = '_'.join([os.path.basename(os.path.normpath(p)).replace('frames_','') for p in input_paths])
        output_path = 'frames_'+basename

    os.makedirs(output_path, exist_ok=True)

    if overwrite or not exists(output_path):

        first_folder = input_paths[0]
        pngfiles = [f for f in listdir(first_folder) if isfile(join(first_folder, f)) and f.endswith('.png')]
        pngfiles.sort()

        from joblib import Parallel, delayed

        def process(pngfile):
            with timeit('Elapsed time: '):
                stacked_image_array = None
                original_dtype = None

                for path in input_paths:

                    print(f"Reading file: {pngfile} from folder: {path}")
                    try:
                        if isfile(path):
                            image_array = imageio.imread(path)
                        else:
                            image_array = imageio.imread(join(path, pngfile))
                    except FileNotFoundError:
                        print(f"WARNING: file {join(path, pngfile)} (or {path}) not found! using blank frame instead!")
                        image_array = numpy.zeros_like(stacked_image_array)

                    if stacked_image_array is None:
                        original_dtype = image_array.dtype
                        stacked_image_array = image_array.astype(numpy.float32)

                    else:
                        if orientation == 'horiz':
                            print(f"Stacks frames horizontally.")
                            stacked_image_array = numpy.hstack((stacked_image_array, image_array))
                        elif orientation == 'vert':
                            print(f"Stacks frames vertically.")
                            stacked_image_array = numpy.vstack((stacked_image_array, image_array))

                print(f"Writing file: {pngfile} in folder: {output_path}")
                imageio.imwrite(join(output_path, pngfile), stacked_image_array.astype(original_dtype))

        Parallel(n_jobs=workers)(delayed(process)(pngfile) for pngfile in pngfiles)


    else:
        print(f"Folder: {output_path} already exists! use -w option to force overwrite...")



cli.add_command(copy)
cli.add_command(add)
cli.add_command(fuse)
cli.add_command(deconv)
cli.add_command(isonet)
cli.add_command(info)
cli.add_command(tiff)
cli.add_command(view)
cli.add_command(render)
cli.add_command(blend)
cli.add_command(stack)
cli.add_command(mp4)
