import glob
from functools import reduce

import click
from arbol.arbol import aprint, asection

from dexp.cli.main import _default_store, _default_codec, _default_clevel, _default_workers_backend
from dexp.cli.utils import _get_dataset_from_path, _parse_channels, _get_output_path
from dexp.datasets.operations.concat import dataset_concat


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ')
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, if -1 then num workers = num channels', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
def concat(input_paths, output_path, channels, store, codec, clevel, overwrite, workers, workersbackend):
    # Make it possible to pass file/folder patterns:
    input_paths = tuple(glob.glob(input_path) for input_path in input_paths)
    input_paths = reduce(lambda u, v: u + v, input_paths)

    input_datasets = tuple(_get_dataset_from_path(input_path) for input_path in input_paths)
    output_path = _get_output_path(input_paths[0], output_path, '_concat')
    channels = _parse_channels(input_datasets[0], channels)

    with asection(f"Concatenating channels: {channels} from: {len(input_datasets)} datasets"):
        for input_path, input_dataset in zip(input_paths, input_datasets):
            aprint(f"dataset at path: {input_path} with channels: {input_dataset.channels()}")

        dataset_concat(channels, input_datasets, output_path, overwrite, store, codec, clevel, workers, workersbackend)

        for input_dataset in input_datasets:
            input_dataset.close()
        aprint("Done!")
