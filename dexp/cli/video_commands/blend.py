import os
from os import listdir
from os.path import exists, isfile, join, isdir

import click
import imageio
import numpy
from arbol.arbol import asection, aprint
from joblib import Parallel, delayed
from scipy import special

from dexp.cli.dexp_main import _default_workers_backend


@click.command()
@click.argument('input_paths', nargs=-1)
@click.option('--output_path', '-o', type=str, default=None, help='Output folder for blended frames.')
@click.option('--blending', '-b', type=str, default='max', help='Blending mode: max, add, adderf, alpha. In the add mode the values are clipped. For alpha-blending, we assume that RGB values are not ‘alpha-premultiplied’.',
              show_default=True)
@click.option('--scales', '-s', type=str, default=None,
              help='List of scales ‘s’ for each input image, starting with the second one (first image has always a scale of 1). Scaled images must be of same dimension or smaller than the first images. ',
              show_default=True)
@click.option('--translations', '-t', type=str, default=None, help='List of translations ‘x,y’ for each input image, starting with the second one (first image is always at 0,0). ',
              show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Force overwrite of output images.', show_default=True)
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, set to -1 for maximum number of workers', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
def blend(input_paths,
          output_path,
          blending,
          scales,
          translations,
          overwrite,
          workers,
          workersbackend):
    if output_path is None:
        basename = '_'.join([os.path.basename(os.path.normpath(p)).replace('frames_', '') for p in input_paths])
        output_path = 'frames_' + basename

    # ensure folder exists:
    os.makedirs(output_path, exist_ok=True)

    # collect all image files:
    image_sequences = []
    for input_path in input_paths:

        if isdir(input_path):
            # path is folder:
            pngfiles = [f for f in listdir(input_path) if isfile(join(input_path, f)) and f.endswith('.png')]
            pngfiles.sort()

        elif isfile(input_path) and (input_path.endswith('png') or input_path.endswith('jpg')):
            # path is image file:
            pngfiles = [input_path, ]

        image_sequences.append(pngfiles)

    # Basic sanity check on the images sequence: we determine the shortest non-one length :
    min_length = min(len(image_sequence) for image_sequence in image_sequences if len(image_sequence) != 1)
    max_length = max(len(image_sequence) for image_sequence in image_sequences if len(image_sequence) != 1)
    if min_length != max_length:
        aprint(f"Not all image sequences have the same non-one length: min:{min_length}, max:{max_length}")

    # Now we broadcast and crop:
    broadcasted_image_sequences = []
    for image_sequence in image_sequences:
        if len(image_sequence) == 1:
            image_sequence = [image_sequence[0], ] * min_length
        elif len(image_sequence) > min_length:
            image_sequence = image_sequence[0:min_length]
        broadcasted_image_sequences.append(image_sequence)

    nb_timepoints = min_length

    with asection(f"Blending  {input_paths}, saving to {output_path}, mode: {blending}, for a total of {nb_timepoints} time points"):
        if workers <= 0:
            workers = os.cpu_count() // 2

        Parallel(n_jobs=workers, backend=workersbackend)(delayed(_process)(broadcasted_image_sequences, tp) for tp in range(nb_timepoints))
        aprint(f"Done!")


def _blend_(image_sequences,
            tp: int):
    with asection(f'processing file: {pngfile}'):
        blended_image_array = None
        original_dtype = None

        for path in input_paths:

            aprint(f"Reading file: {pngfile} from path: {path}")
            try:
                if isfile(path):
                    image_array = imageio.imread(path)
                else:
                    image_array = imageio.imread(join(path, pngfile))
            except FileNotFoundError:
                aprint(f"WARNING: file {join(path, pngfile)} (or {path}) not found! using blank frame instead!")
                image_array = numpy.zeros_like(blended_image_array)

            # convert image to floating point:
            image_array = image_array.astype(numpy.float32)

            # normalise image:
            image_array /= 255

            if blended_image_array is None:
                # If no other image has been blended yet, we copy this image to the current blended image:
                original_dtype = image_array.dtype
                blended_image_array = image_array.astype(numpy.float32)

            else:
                # Otherwise we blend:
                if blending == 'max':
                    aprint(f"Blending using max mode.")
                    blended_image_array = numpy.maximum(blended_image_array, image_array)
                elif blending == 'add' or blending == 'addclip':
                    aprint(f"Blending using add mode (clipped).")
                    blended_image_array = blended_image_array + image_array
                elif blending == 'adderf':
                    aprint(f"Blending using add mode (erf saturation).")
                    blended_image_array = special.erf(blended_image_array + image_array)
                elif blending == 'alpha' and blended_image_array.shape[-1] == 4 and image_array.shape[-1] == 4:
                    aprint(f"Blending using alpha mode.")
                    # See: https://en.wikipedia.org/wiki/Alpha_compositing

                    src_rgb = image_array[..., 0:3]
                    src_alpha = image_array[..., 3]
                    dst_rgb = blended_image_array[..., 0:3]
                    dst_alpha = blended_image_array[..., 3]

                    out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
                    out_rgb = (src_rgb * src_alpha + dst_rgb * dst_alpha * (1 - src_alpha)) / out_alpha

                    blended_image_array[..., 0:3] = out_rgb
                    blended_image_array[..., 3] = out_alpha

                else:
                    raise ValueError(f"Invalid blending mode or incompatible images: shape={image_array.shape}")

        # convert back to 255 range, clip, and adjust dtype:
        blended_image_array *= 255
        blended_image_array = numpy.clip(blended_image_array, 0, 255)
        blended_image_array = blended_image_array.astype(original_dtype, copy=False)

        # write file:
        file_path = join(output_path, pngfile)
        if overwrite or not exists(file_path):
            aprint(f"Writing file: {pngfile} in folder: {output_path}")
            imageio.imwrite(file_path, blended_image_array)
        else:
            aprint(f"File: {file_path} already exists! use -w option to force overwrite...")
