from time import time

import click

from dexp.cli.main import _get_dataset_from_path, _get_folder_name_without_end_slash, _default_clevel
from dexp.datasets.operations.tiff import dataset_tiff


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


    print(f"Saving dataset to TIFF file: {output_path}")
    time_start = time()
    dataset_tiff(   input_dataset,
                    output_path,
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
    input_dataset.close()
    print("Done!")
