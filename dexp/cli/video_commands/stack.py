import os
from os import listdir
from os.path import exists, isfile, join

import click
import imageio
import numpy
from arbol.arbol import aprint, asection
from joblib import Parallel, delayed

from dexp.cli.dexp_main import _default_workers_backend


@click.command()
@click.argument('input_paths', nargs=-1)
@click.option('--output_path', '-o', type=str, default=None, help='Output folder for blended frames.')
@click.option('--orientation', '-r', type=str, default='horiz', help='Stitching mode: horiz, vert', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Force overwrite of output images.', show_default=True)
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, set to -1 for maximum number of workers', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)
def stack(input_paths, output_path, orientation, overwrite, workers, workersbackend):
    if workers <= 0:
        workers = os.cpu_count() // 2

    if output_path is None:
        basename = '_'.join([os.path.basename(os.path.normpath(p)).replace('frames_', '') for p in input_paths])
        output_path = 'frames_' + basename

    os.makedirs(output_path, exist_ok=True)

    first_folder = input_paths[0]
    pngfiles = [f for f in listdir(first_folder) if isfile(join(first_folder, f)) and f.endswith('.png')]
    pngfiles.sort()

    def process(pngfile):
        stacked_image_array = None
        original_dtype = None

        for path in input_paths:
            aprint(f"Reading file: {pngfile} from folder: {path}")
            try:
                if isfile(path):
                    image_array = imageio.imread(path)
                else:
                    image_array = imageio.imread(join(path, pngfile))
            except FileNotFoundError:
                aprint(f"WARNING: file {join(path, pngfile)} (or {path}) not found! using blank frame instead!")
                image_array = numpy.zeros_like(stacked_image_array)

            if stacked_image_array is None:
                original_dtype = image_array.dtype
                stacked_image_array = image_array.astype(numpy.float32)

            else:
                if orientation == 'horiz':
                    aprint(f"Stacks frames horizontally.")
                    stacked_image_array = numpy.hstack((stacked_image_array, image_array))
                elif orientation == 'vert':
                    aprint(f"Stacks frames vertically.")
                    stacked_image_array = numpy.vstack((stacked_image_array, image_array))

        stacked_image_array = stacked_image_array.astype(original_dtype)

        # write file:
        file_path = join(output_path, pngfile)
        if overwrite or not exists(file_path):
            aprint(f"Writing file: {pngfile} in folder: {output_path}")
            imageio.imwrite(file_path, stacked_image_array)
        else:
            aprint(f"File: {file_path} already exists! use -w option to force overwrite...")

    with asection(f"stacking PNGs: {input_paths}, saving to: {output_path}, orientation: {orientation}"):
        Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(pngfile) for pngfile in pngfiles)
        aprint("Done!")
