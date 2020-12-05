import click

from dexp.cli.main import _default_codec, _default_store, _default_clevel
from dexp.cli.utils import _parse_channels, _get_dataset_from_path, _get_output_path, _parse_slicing, _parse_devices
from dexp.datasets.operations.fuse import dataset_fuse
from dexp.utils.timeit import timeit


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='list of channels for the view in standard order for the microscope type (C0L0, C0L1, C1L0, C1L1,...)')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ')
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--microscope', '-m', type=str, default='simview', help='Microscope objective to use for computing psf, can be: simview or mvsols', show_default=True)
@click.option('--equalise/--no-equalise', '-eq', default=True, help='Equalise intensity of views before fusion, or not.', show_default=True)
@click.option('--zerolevel', '-zl', type=int, default=110, help='\'zero-level\' i.e. the pixel values in the restoration (to be substracted)', show_default=True)  #
@click.option('--fusion', '-f', type=str, default='tg', help="Fusion mode, can be: 'tg' or 'dct'.  ", show_default=True)  #
@click.option('--fusion_bias_strength', '-fbs', type=float, default=0.1, help='Fusion bias strength, set to 0 if fusing a cropped region', show_default=True)  #
@click.option('--dehaze_size', '-dhs', type=int, default=65, help='Filter size (scale) for dehazing the final regsitered and fused image to reduce effect of scattered and out-of-focus light. Set to zero to deactivate.',
              show_default=True)  #
@click.option('--dark_denoise_threshold', '-ddt', type=int, default=0, help='Threshold for denoises the dark pixels of the image -- helps increase compression ratio. Set to zero to deactivate.', show_default=True)  #
@click.option('--loadshifts', '-ls', is_flag=True, help='Turn on to load the registration parameters (i.e translation shifts) from another run', show_default=True)  #
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)  #
@click.option('--devices', '-d', type=str, default='0', help='Sets the CUDA devices id, e.g. 0,1,2', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def fuse(input_path,
         output_path,
         channels,
         slicing,
         store,
         codec,
         clevel,
         overwrite,
         microscope,
         equalise,
         zerolevel,
         fusion,
         fusion_bias_strength,
         dehaze_size,
         dark_denoise_threshold,
         loadshifts,
         workers,
         devices,
         check):
    input_dataset = _get_dataset_from_path(input_path)
    output_path = _get_output_path(input_path, output_path, ".fused")

    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)
    devices = _parse_devices(devices)

    with timeit(f""):
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
                     zero_level=zerolevel,
                     fusion=fusion,
                     fusion_bias_strength=fusion_bias_strength,
                     dehaze_size=dehaze_size,
                     dark_denoise_threshold=dark_denoise_threshold,
                     load_shifts=loadshifts,
                     workers=workers,
                     devices=devices,
                     check=check,
                     )
    input_dataset.close()
    print("Done!")
