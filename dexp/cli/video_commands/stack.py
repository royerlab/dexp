import os
from os import listdir
from os.path import exists, isdir, isfile, join

import click
import imageio
import numpy
from arbol.arbol import aprint, asection
from joblib import Parallel, delayed

from dexp.cli.defaults import DEFAULT_WORKERS_BACKEND
from dexp.utils.backends import Backend


@click.command()
@click.argument("input_paths", nargs=-1)
@click.option("--output_path", "-o", type=str, default=None, help="Output folder for stacked frames.")
@click.option("--orientation", "-or", type=str, default="horiz", help="Stitching mode: horiz, vert", show_default=True)
@click.option("--overwrite", "-w", is_flag=True, help="Force overwrite of output images.", show_default=True)
@click.option(
    "--workers",
    "-k",
    type=int,
    default=-1,
    help="Number of worker threads to spawn, set to -1 for maximum number of workers",
    show_default=True,
)  #
@click.option(
    "--workersbackend",
    "-wkb",
    type=str,
    default=DEFAULT_WORKERS_BACKEND,
    help="What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ",
    show_default=True,
)
def stack(input_paths, output_path, orientation, overwrite, workers, workersbackend):
    """Stacks video vertically or horizontally (video = folder of images)."""

    if workers <= 0:
        workers = os.cpu_count() // 2

    if output_path is None:
        output_path = input_paths[0] + "_stack"
    elif output_path.startswith("_"):
        output_path = input_paths[0] + output_path

    os.makedirs(output_path, exist_ok=True)

    # collect all image files:
    image_sequences = []
    for input_path in input_paths:

        if isdir(input_path):
            # path is folder:
            pngfiles = [
                join(input_path, f) for f in listdir(input_path) if isfile(join(input_path, f)) and f.endswith(".png")
            ]
            pngfiles.sort()

        elif isfile(input_path) and (input_path.endswith("png") or input_path.endswith("jpg")):
            # path is image file:
            pngfiles = [
                input_path,
            ]

        image_sequences.append(pngfiles)

    # Basic sanity check on the images sequence: we determine the shortest non-one length :
    min_length = min(len(image_sequence) for image_sequence in image_sequences if len(image_sequence) != 1)
    max_length = max(len(image_sequence) for image_sequence in image_sequences if len(image_sequence) != 1)
    if min_length != max_length:
        aprint(f"Not all image sequences have the same non-one length: min:{min_length}, max:{max_length}")

    # Now we broadcast and crop in time:
    _image_sequences = []
    for image_sequence in image_sequences:
        if len(image_sequence) == 1:
            image_sequence = [
                image_sequence[0],
            ] * min_length
        elif len(image_sequence) > min_length:
            image_sequence = image_sequence[0:min_length]
        _image_sequences.append(image_sequence)
    image_sequences = _image_sequences
    nb_timepoints = min_length

    def process(tp):

        # Output file:
        filename = f"frame_{tp:05}.png"
        filepath = join(output_path, filename)

        # Write file:
        if overwrite or not exists(filepath):

            stacked_image_array = None

            # collect all images that need to be blended:
            image_paths = list(image_sequence[tp] for image_sequence in image_sequences)

            # Load images:
            images = list(imageio.imread(image_path) for image_path in image_paths)

            if "horiz" in orientation:
                aprint("Stacks frames horizontally.")
                stacked_image_array = numpy.hstack(images)
            elif "vert" in orientation:
                aprint("Stacks frames vertically.")
                stacked_image_array = numpy.vstack(images)

            aprint(f"Writing file: {filename} in folder: {output_path}")
            imageio.imwrite(filepath, Backend.to_numpy(stacked_image_array), compress_level=1)
        else:
            aprint(f"File: {output_path} already exists! use -w option to force overwrite...")

    with asection(f"stacking PNGs: {input_paths}, saving to: {output_path}, orientation: {orientation}"):
        Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp) for tp in range(nb_timepoints))
        aprint("Done!")
