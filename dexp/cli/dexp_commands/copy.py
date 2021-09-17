import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import _default_clevel, _default_codec, _default_store, _default_workers_backend
from dexp.cli.parsing import _parse_channels, _get_output_path, _parse_slicing, _parse_chunks
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.copy import dataset_copy


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='Dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--chunks', '-chk', default=None, help='Dataset chunks dimensions, e.g. (1, 126, 512, 512).')
@click.option('--codec', '-z', default=_default_codec, help='Compression codec: zstd for ’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
@click.option('--zerolevel', '-zl', type=int, default=0, help="‘zero-level’ i.e. the pixel values in the restoration (to be substracted)", show_default=True)  #
@click.option('--workers', '-wk', default=-4, help='Number of worker threads to spawn. Negative numbers n correspond to: number_of _cores / |n| ', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def copy(input_paths,
         output_path,
         channels,
         slicing,
         store,
         chunks,
         codec,
         clevel,
         overwrite,
         zerolevel,
         workers,
         workersbackend,
         check):
    """ Copies a dataset, channels can be selected, cropping can be performed, compression can be changed, ...
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path, '_copy')
    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)
    chunks = _parse_chunks(chunks)

    with asection(f"Copying from: {input_paths} to {output_path} for channels: {channels}, slicing: {slicing} "):
        dataset_copy(input_dataset,
                     output_path,
                     channels=channels,
                     slicing=slicing,
                     store=store,
                     chunks=chunks,
                     compression=codec,
                     compression_level=clevel,
                     overwrite=overwrite,
                     zerolevel=zerolevel,
                     workers=workers,
                     workersbackend=workersbackend,
                     check=check)

        input_dataset.close()
        aprint("Done!")
