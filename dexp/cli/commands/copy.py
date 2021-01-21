import click
from arbol.arbol import aprint, asection

from dexp.cli.main import _default_clevel, _default_codec, _default_store
from dexp.cli.main import _default_workers_backend
from dexp.cli.utils import _parse_channels, _get_dataset_from_path, _get_output_path, _parse_slicing
from dexp.datasets.operations.copy import dataset_copy


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='Dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='Compression codec: zstd for ’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
@click.option('--workers', '-wk', default=-1, help='Number of worker threads to spawn.', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def copy(input_path, output_path, channels, slicing, store, codec, clevel, overwrite, workers, workersbackend, check):
    input_dataset = _get_dataset_from_path(input_path)
    output_path = _get_output_path(input_path, output_path, '_copy')
    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)

    with asection(f"Copying from: {input_path} to {output_path} for channels: {channels}, slicing: {slicing} "):
        dataset_copy(input_dataset,
                     output_path,
                     channels=channels,
                     slicing=slicing,
                     store=store,
                     compression=codec,
                     compression_level=clevel,
                     overwrite=overwrite,
                     workers=workers,
                     workersbackend=workersbackend,
                     check=check)

        input_dataset.close()
        aprint("Done!")
