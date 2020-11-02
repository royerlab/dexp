from time import time

import click

from dexp.cli.main import _get_dataset_from_path, _default_codec, _default_store, _default_clevel, _parse_slicing, _get_folder_name_without_end_slash
from dexp.datasets.operations.fuse import dataset_fuse


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--slicing', '-s', default=None , help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--store', '-st', default=_default_store, help='Store: ‘dir’, ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ')
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--workers', '-k', type=int, default=1, help='Number of worker threads to spawn, recommended: 1 (unless you know what you are doing)', show_default=True)  #
@click.option('--zerolevel', '-zl', type=int, default=110,  help='Sets the \'zero-level\' i.e. the pixel values in the restoration (to be substracted)', show_default=True)  #
@click.option('--loadshifts', '-ls', is_flag=True, help='Turn on to load the registration parameters (i.e translation shifts) from another run', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def fuse(input_path, output_path, slicing, store, codec, clevel, overwrite, workers, zerolevel, loadshifts, check):
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
    dataset_fuse(input_dataset,
         output_path,
         slicing=slicing,
         store=store,
         compression=codec,
         compression_level=clevel,
         overwrite=overwrite,
         workers=workers,
         zero_level=zerolevel,
         load_shifts=loadshifts,
         check=check
         )
    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    input_dataset.close()
    print("Done!")
    pass
