import os

import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import _get_output_path
from dexp.utils.robocopy import robocopy


@click.command()
@click.argument('input_path', nargs=1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--workers', '-wk', default=8, help='Number of worker threads to spawn. Negative numbers n correspond to: number_of _cores / |n| ', show_default=True)  #
@click.option('--large_files', '-lf', is_flag=True, help='Set to true to speed up large file transfer', show_default=True)
def fastcopy(input_path,
             output_path,
             workers,
             large_files):
    """ Copies a dataset fast, with no processing, just moves the data as fast as possible. For each operating system it uses the best method.
    """

    output_path = _get_output_path(input_path, output_path, '_copy')

    if workers == -1:
        workers = max(1, os.cpu_count() // abs(workers))

    with asection(f"Fast copying from: {input_path} to {output_path} "):

        from sys import platform
        if platform == "linux" or platform == "linux2":
            raise NotImplementedError("Fast copy not yet implemented for Linux")
        elif platform == "darwin":
            raise NotImplementedError("Fast copy not yet implemented for OSX")
        elif platform == "win32":
            robocopy(input_path, output_path, nb_threads=workers, large_files=large_files or not('.zarr' in input_path))

        aprint("Done!")
