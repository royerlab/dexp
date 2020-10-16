import os
from os import listdir
from os.path import exists, isfile, join
from timeit import timeit

import click
import imageio
import numpy
from scipy import special


@click.command()
@click.argument('input_paths', nargs=-1)
@click.option('--output_path', '-o', type=str, default=None, help='Output folder for blended frames.')
@click.option('--blending', '-b', type=str, default='max', help='Blending mode: max, add, addclip, adderf (add stands for addclip).', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, set to -1 for maximum number of workers', show_default=True)  #
def blend(input_paths, output_path, blending, overwrite, workers):

    if workers <= 0:
        workers = os.cpu_count()//2

    if output_path is None:
        basename = '_'.join([os.path.basename(os.path.normpath(p)).replace('frames_','') for p in input_paths])
        output_path = 'frames_'+basename

    os.makedirs(output_path, exist_ok=True)

    if overwrite or not exists(output_path):

        first_folder = input_paths[0]
        pngfiles = [f for f in listdir(first_folder) if isfile(join(first_folder, f)) and f.endswith('.png')]
        pngfiles.sort()

        from joblib import Parallel, delayed

        def process(pngfile):
            with timeit('Elapsed time: '):
                blended_image_array = None
                original_dtype = None

                for path in input_paths:

                    print(f"Reading file: {pngfile} from path: {path}")
                    try:
                        if isfile(path):
                            image_array = imageio.imread(path)
                        else:
                            image_array = imageio.imread(join(path, pngfile))
                    except FileNotFoundError:
                        print(f"WARNING: file {join(path, pngfile)} (or {path}) not found! using blank frame instead!")
                        image_array = numpy.zeros_like(blended_image_array)

                    if blended_image_array is None:
                        original_dtype = image_array.dtype
                        blended_image_array = image_array.astype(numpy.float32)

                    else:
                        if blending == 'max':
                            print(f"Blending using max mode.")
                            blended_image_array = numpy.maximum(blended_image_array, image_array)
                        elif blending == 'add' or  blending == 'addclip':
                            print(f"Blending using add mode (clipped).")
                            blended_image_array = numpy.clip(blended_image_array+image_array, 0, 255)
                        elif blending == 'adderf':
                            print(f"Blending using add mode (erf saturation).")
                            blended_image_array = 255*special.erf(blended_image_array+image_array/255)

                print(f"Writing file: {pngfile} in folder: {output_path}")
                imageio.imwrite(join(output_path, pngfile), blended_image_array.astype(original_dtype))

        Parallel(n_jobs=workers)(delayed(process)(pngfile) for pngfile in pngfiles)


    else:
        print(f"Folder: {output_path} already exists! use -w option to force overwrite...")