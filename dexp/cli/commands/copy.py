from time import time

import click

from dexp.cli.main import _get_dataset_from_path, _get_folder_name_without_end_slash, _parse_slicing, _default_clevel, _default_codec, _default_store
from dexp.datasets.operations.copy import dataset_copy


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='Dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--store', '-st', default=_default_store, help='Store: ‘dir’, ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='Compression codec: zstd for ’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
@click.option('--project', '-p', type=int, default=None, help='max projection over given axis (0->T, 1->Z, 2->Y, 3->X)')
@click.option('--workers', '-k', default=1, help='Number of worker threads to spawn.', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def copy(input_path, output_path, channels, slicing, store, codec, clevel, overwrite, project, workers, check):
    input_dataset = _get_dataset_from_path(input_path)

    print(f"Available Channel(s): {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = _get_folder_name_without_end_slash(input_path) + '.zarr'

    slicing = _parse_slicing(slicing)
    print(f"Requested slicing: {slicing} ")

    print(f"Requested channel(s)  {channels if channels else '--All--'} ")
    if not channels is None:
        channels = channels.split(',')
    print(f"Selected channel(s): '{channels}' and slice: {slicing}")

    print("Converting dataset.")
    print(f"Saving dataset to: {output_path} with zarr format... ")
    time_start = time()
    dataset_copy(input_dataset,
                 output_path,
                 channels=channels,
                 slicing=slicing,
                 store=store,
                 compression=codec,
                 compression_level=clevel,
                 overwrite=overwrite,
                 project=project,
                 workers=workers,
                 check=check)
    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    input_dataset.close()
    print("Done!")
