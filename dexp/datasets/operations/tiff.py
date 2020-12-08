import os
from os.path import join

import numpy
from arbol.arbol import aprint
from tifffile import TiffWriter

from dexp.io.io import tiff_save
from dexp.utils.timeit import timeit


def dataset_tiff(dataset,
                 output_path,
                 channels,
                 slicing,
                 overwrite,
                 project,
                 one_file_per_first_dim,
                 clevel,
                 workers):
    selected_channels = dataset._selected_channels(channels)

    aprint(f"getting Dask arrays for channels {selected_channels}")
    arrays = list([dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True) for channel in selected_channels])

    if slicing is not None:
        aprint(f"Slicing with: {slicing}")
        arrays = list([array[slicing] for array in arrays])
        aprint(f"Done slicing.")

    if project:
        # project is the axis for projection, but here we are not considering the T dimension anymore...
        aprint(f"Projecting along axis {project}")
        arrays = list([array.max(axis=project) for array in arrays])

    if one_file_per_first_dim:
        aprint(f"Saving one TIFF file for each tp (or Z if already sliced) to: {output_path}.")

        os.makedirs(output_path, exist_ok=True)

        from joblib import Parallel, delayed

        def process(tp):
            with timeit('Elapsed time: '):
                for channel, array in zip(selected_channels, arrays):
                    tiff_file_path = join(output_path, f"file{tp}_{channel}.tiff")
                    if overwrite or not os.path.exists(tiff_file_path):
                        stack = array[tp].compute()
                        print(f"Writing time point: {tp} of shape: {stack.shape}, dtype:{stack.dtype} as TIFF file: '{tiff_file_path}', with compression: {clevel}")
                        tiff_save(tiff_file_path, stack, compress=clevel)
                        print(f"Done writing time point: {tp} !")
                    else:
                        print(f"File for time point (or z slice): {tp} already exists.")

        Parallel(n_jobs=workers)(delayed(process)(tp) for tp in range(0, arrays[0].shape[0]))


    else:
        array = numpy.stack(arrays)

        if not overwrite and os.path.exists(output_path):
            aprint(f"File {output_path} already exists! Set option -w to overwrite.")
            return

        aprint(f"Creating memory mapped TIFF file at: {output_path}.")
        with TiffWriter(output_path, bigtiff=True, imagej=True) as tif:
            tp = 0
            for stack in array:
                with timeit('Elapsed time: '):
                    aprint(f"Writing time point: {tp} ")
                    stack = stack.compute()
                    tif.save(stack)
                    tp += 1
