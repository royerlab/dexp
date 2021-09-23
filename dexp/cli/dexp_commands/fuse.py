import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import _default_codec, _default_store, _default_clevel, _default_workers_backend
from dexp.cli.parsing import _parse_channels, _get_output_path, _parse_slicing, _parse_devices
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.fuse import dataset_fuse


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='list of channels for the view in standard order for the microscope type (C0L0, C0L1, C1L0, C1L1,...)')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ')
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--microscope', '-m', type=str, default='simview', help='Microscope objective to use for computing psf, can be: simview or mvsols', show_default=True)
@click.option('--equalise/--no-equalise', '-eq/-neq', default=True, help='Equalise intensity of views before fusion, or not.', show_default=True)
@click.option('--equalisemode', '-eqm', default='first', help='Equalisation modes: compute correction ratios only for first time point: ‘first’ or for all time points: ‘all’.', show_default=True)
@click.option('--zerolevel', '-zl', type=int, default=0, help="‘zero-level’ i.e. the pixel values in the restoration (to be substracted)", show_default=True)  #
@click.option('--cliphigh', '-ch', type=int, default=0, help='Clips voxel values above the given value, if zero no clipping is done', show_default=True)  #
@click.option('--fusion', '-f', type=str, default='tg', help="Fusion mode, can be: ‘tg’ or ‘dct’.  ", show_default=True)  #
@click.option('--fusion_bias_strength', '-fbs', type=(float, float), default=(0.5, 0.02), help='Fusion bias strength for illumination and detection ‘fbs_i fbs_d’, set to ‘0 0’) if fusing a cropped region', show_default=True)  #
@click.option('--dehaze_size', '-dhs', type=int, default=65, help='Filter size (scale) for dehazing the final regsitered and fused image to reduce effect of scattered and out-of-focus light. Set to zero to deactivate.',
              show_default=True)  #
@click.option('--dark_denoise_threshold', '-ddt', type=int, default=0, help='Threshold for denoises the dark pixels of the image -- helps increase compression ratio. Set to zero to deactivate.', show_default=True)  #
@click.option('--zpadapodise', '-zpa', type=(int, int), default=(8, 96),
              help='Pads and apodises the views along z before fusion: ‘pad apo’, where pad is a padding length, and apo is apodisation length, both in voxels. If pad=apo, no original voxel is modified and only added voxels are apodised.',
              show_default=True)  #
@click.option('--loadreg', '-lr', is_flag=True, help='Turn on to load the registration parameters from a previous run', show_default=True)  #
@click.option('--model-filename', '-mf', help='Model filename to load or save registration model list', default='registration_models.txt', show_default=True) 
@click.option('--warpregiter', '-wri', type=int, default=4, help='Number of iterations for warp registration (if applicable).', show_default=True)  #
@click.option('--minconfidence', '-mc', type=float, default=0.3, help='Minimal confidence for registration parameters, if below that level the registration parameters for previous time points is used.', show_default=True)  #
@click.option('--maxchange', '-md', type=float, default=16, help='Maximal change in registration parameters, if above that level the registration parameters for previous time points is used.', show_default=True)  #
@click.option('--regedgefilter', '-ref', is_flag=True, help='Use this flag to apply an edge filter to help registration.', show_default=True)  #
@click.option('--maxproj/--no-maxproj', '-mp/-nmp', type=bool, default=True, help='Registers using only the maximum intensity projection from each stack.', show_default=True)
@click.option('--hugedataset', '-hd', is_flag=True, help='Use this flag to indicate that the the dataset is _huge_ and that memory allocation should be optimised at the detriment of processing speed.', show_default=True)  #
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
@click.option('--devices', '-d', type=str, default='0', help='Sets the CUDA devices id, e.g. 0,1,2 or ‘all’', show_default=True)  #
@click.option('--pad', '-p', is_flag=True, default=False, help='Use this flag to pad views according to the registration models.')
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def fuse(input_paths,
         output_path,
         channels,
         slicing,
         store,
         codec,
         clevel,
         overwrite,
         microscope,
         equalise,
         equalisemode,
         zerolevel,
         cliphigh,
         fusion,
         fusion_bias_strength,
         dehaze_size,
         dark_denoise_threshold,
         zpadapodise,
         loadreg,
         model_filename,
         warpregiter,
         minconfidence,
         maxchange,
         regedgefilter,
         maxproj,
         hugedataset,
         workers,
         workersbackend,
         devices,
         pad,
         check):
    """ Fuses the views of a multi-view light-sheet microscope dataset (available: simview and mvsols)
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path, "_fused")

    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)
    devices = _parse_devices(devices)

    with asection(f"Fusing dataset: {input_paths}, saving it at: {output_path}, for channels: {channels}, slicing: {slicing} "):
        aprint(f"Microscope type: {microscope}, fusion type: {fusion}")
        aprint(f"Devices used: {devices}, workers: {workers} ")
        dataset_fuse(input_dataset,
                     output_path,
                     channels=channels,
                     slicing=slicing,
                     store=store,
                     compression=codec,
                     compression_level=clevel,
                     overwrite=overwrite,
                     microscope=microscope,
                     equalise=equalise,
                     equalise_mode=equalisemode,
                     zero_level=zerolevel,
                     clip_too_high=cliphigh,
                     fusion=fusion,
                     fusion_bias_strength_i=fusion_bias_strength[0],
                     fusion_bias_strength_d=fusion_bias_strength[1],
                     dehaze_size=dehaze_size,
                     dark_denoise_threshold=dark_denoise_threshold,
                     z_pad_apodise=zpadapodise,
                     loadreg=loadreg,
                     model_list_filename=model_filename,
                     warpreg_num_iterations=warpregiter,
                     min_confidence=minconfidence,
                     max_change=maxchange,
                     registration_edge_filter=regedgefilter,
                     maxproj=maxproj,
                     huge_dataset=hugedataset,
                     workers=workers,
                     workersbackend=workersbackend,
                     devices=devices,
                     pad=pad,
                     check=check)

        input_dataset.close()
        aprint("Done!")
