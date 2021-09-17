import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import _default_workers_backend
from dexp.cli.parsing import _parse_channels, _get_output_path, _parse_slicing
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.tiff import dataset_tiff


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='selected channels.')  #
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--project', '-p', type=int, default=None, help='max projection over given axis (0->T, 1->Z, 2->Y, 3->X)')  # , help='dataset slice'
@click.option('--split', is_flag=True, help='Splits dataset along first dimension, be carefull, if you slice to a single time point this will split along z!')  # , help='dataset slice'
@click.option('--clevel', '-l', type=int, default=0, help='Compression level, 0 means no compression, max is 9', show_default=True)  # , help='dataset slice'
@click.option('--workers', '-k', default=-4, help='Number of worker threads to spawn. Negative numbers n correspond to: number_of _cores / |n|', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)
def tiff(input_paths,
         output_path,
         channels,
         slicing,
         overwrite,
         project,
         split,
         clevel,
         workers,
         workersbackend):
    """ Exports dataset as TIFF file(s).
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path)
    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)

    with asection(f"Exporting to TIFF datset: {input_paths}, channels: {channels}, slice: {slicing}, project:{project}, split:{split}"):
        dataset_tiff(input_dataset,
                     output_path,
                     channels=channels,
                     slicing=slicing,
                     overwrite=overwrite,
                     project=project,
                     one_file_per_first_dim=split,
                     clevel=clevel,
                     workers=workers,
                     workersbackend=workersbackend
                     )

        input_dataset.close()
        aprint("Done!")
