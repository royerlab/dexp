import os
from os import listdir
from os.path import exists, isfile, join

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
@click.option('--blending', '-b', type=str, default='max', help='Blending mode: max, add, adderf, alpha. Note: in the add mode the values are clipped.', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, set to -1 for maximum number of workers', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
def blend(input_paths, output_path, blending, overwrite, workers, workersbackend):
    if workers <= 0:
        workers = os.cpu_count() // 2

    if output_path is None:
        basename = '_'.join([os.path.basename(os.path.normpath(p)).replace('frames_', '') for p in input_paths])
        output_path = 'frames_' + basename

    os.makedirs(output_path, exist_ok=True)

    if overwrite or not exists(output_path):

        first_folder = input_paths[0]
        pngfiles = [f for f in listdir(first_folder) if isfile(join(first_folder, f)) and f.endswith('.png')]
        pngfiles.sort()

        def process(pngfile):
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
                            blended_image_array =blended_image_array + image_array
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
                            out_rgb = (src_rgb*src_alpha + dst_rgb*dst_alpha*(1-src_alpha)) / out_alpha

                            blended_image_array[..., 0:3] = out_rgb
                            blended_image_array[..., 3] = out_alpha

                        else:
                            raise ValueError(f"Invalid blending mode or incompatible images: shape={image_array.shape}")

                # convert back to 255 range, clip, and adjust dtype:
                blended_image_array *= 255
                blended_image_array = numpy.clip(blended_image_array, 0, 255)
                blended_image_array = blended_image_array.astype(original_dtype, copy=False)

                # write file:
                aprint(f"Writing file: {pngfile} in folder: {output_path}")
                imageio.imwrite(join(output_path, pngfile), blended_image_array)

        with asection(f"Blending  {input_paths}, saving to {output_path}, mode: {blending}"):
            Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(pngfile) for pngfile in pngfiles)
            aprint(f"Done!")

    else:
        aprint(f"Folder: {output_path} already exists! use -w option to force overwrite...")
