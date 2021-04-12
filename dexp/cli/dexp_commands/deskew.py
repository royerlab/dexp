import click
from arbol.arbol import aprint, asection

from dexp.cli.dexp_main import _default_codec, _default_store, _default_clevel
from dexp.cli.dexp_main import _default_workers_backend
from dexp.cli.parsing import _parse_channels, _get_output_path, _parse_slicing, _parse_devices
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.deskew import dataset_deskew


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='list of channels for the view in standard order for the microscope type (C0L0, C0L1, C1L0, C1L1,...)')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ')
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--zerolevel', '-zl', type=int, default=110, help="‘zero-level’ i.e. the pixel values in the restoration (to be substracted)", show_default=True)  #
@click.option('--cliphigh', '-ch', type=int, default=1024, help='Clips voxel values above the given value, if zero no clipping is done', show_default=True)  #
@click.option('--dehaze_size', '-dhs', type=int, default=65, help='Filter size (scale) for dehazing the final regsitered and fused image to reduce effect of scattered and out-of-focus light. Set to zero to deactivate.',
              show_default=True)  #
@click.option('--dark_denoise_threshold', '-ddt', type=int, default=0, help='Threshold for denoises the dark pixels of the image -- helps increase compression ratio. Set to zero to deactivate.', show_default=True)  #
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
@click.option('--devices', '-d', type=str, default='0', help='Sets the CUDA devices id, e.g. 0,1,2 or ‘all’', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def deskew(input_paths,
           output_path,
           channels,
           slicing,
           store,
           codec,
           clevel,
           overwrite,
           zerolevel,
           cliphigh,
           dehaze_size,
           dark_denoise_threshold,
           workers,
           workersbackend,
           devices,
           check):
    """ Deskews all or selected channels of a dataset.
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path, "_deconv")

    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)
    devices = _parse_devices(devices)

    with asection(f"Deskewing dataset: {input_paths}, saving it at: {output_path}, for channels: {channels}, slicing: {slicing} "):
        aprint(f"Devices used: {devices}, workers: {workers} ")
        dataset_deskew(input_dataset,
                       output_path,
                       channels=channels,
                       slicing=slicing,
                       store=store,
                       compression=codec,
                       compression_level=clevel,
                       overwrite=overwrite,
                       zero_level=zerolevel,
                       clip_too_high=cliphigh,
                       dehaze_size=dehaze_size,
                       dark_denoise_threshold=dark_denoise_threshold,
                       workers=workers,
                       workersbackend=workersbackend,
                       devices=devices,
                       check=check)

        input_dataset.close()
        aprint("Done!")
