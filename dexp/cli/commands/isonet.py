from time import time

import click

from dexp.cli.main import _get_dataset_from_path, _default_clevel, _default_codec, _default_store, _get_folder_name_without_end_slash, _parse_slicing
from dexp.datasets.operations.isonet import dataset_isonet


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
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def isonet(input_path, output_path, slicing, store, codec, clevel, overwrite, context, mode, max_epochs, check):
    input_dataset = _get_dataset_from_path(input_path)

    print(f"Available Channels: {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = _get_folder_name_without_end_slash(input_path) + '.zarr'

    slicing = _parse_slicing(slicing)
    print(f"Requested slicing: {slicing} ")

    print("Fusing dataset.")
    print(f"Saving dataset to: {output_path} with zarr format... ")
    time_start = time()
    dataset_isonet(input_dataset,
           output_path,
           slicing=slicing,
           store=store,
           compression=codec,
           compression_level=clevel,
           overwrite=overwrite,
           context=context,
           mode=mode,
           max_epochs=max_epochs,
           check=check
           )

    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    input_dataset.close()
    print("Done!")
