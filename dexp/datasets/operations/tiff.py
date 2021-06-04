import os
from os.path import join
from typing import Sequence, Union

from arbol.arbol import aprint, asection
from joblib import Parallel, delayed
from tifffile import memmap

from dexp.datasets.base_dataset import BaseDataset
from dexp.io.io import tiff_save


def dataset_tiff(dataset: BaseDataset,
                 dest_path: str,
                 channels: Sequence[str],
                 slicing,
                 overwrite: bool = False,
                 project: Union[int, bool] = False,
                 one_file_per_first_dim: bool = False,
                 clevel: int = 0,
                 workers: int = 1,
                 workersbackend: str = '',
                 stop_at_exception: bool = True):


    selected_channels = dataset._selected_channels(channels)

    aprint(f"getting Dask arrays for channels {selected_channels}")
    arrays = list([dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True) for channel in selected_channels])

    if slicing is not None:
        aprint(f"Slicing with: {slicing}")
        arrays = list([array[slicing] for array in arrays])
        aprint(f"Done slicing.")

    if workers == -1:
        workers = max(1, os.cpu_count() // abs(workers))
    aprint(f"Number of workers: {workers}")

    if one_file_per_first_dim:
        aprint(f"Saving one TIFF file for each tp (or Z if already sliced) to: {dest_path}.")

        os.makedirs(dest_path, exist_ok=True)

        def process(tp):
            try:
                with asection(f'Saving time point {tp}: '):
                    for channel, array in zip(selected_channels, arrays):
                        tiff_file_path = join(dest_path, f"file{tp}_{channel}.tiff")
                        if overwrite or not os.path.exists(tiff_file_path):
                            stack = array[tp].compute()

                            if project is not False and type(project) == int:
                                # project is the axis for projection, but here we are not considering the T dimension anymore...
                                aprint(f"Projecting along axis {project}")
                                stack = stack.max(axis=project)

                            aprint(f"Writing time point: {tp} of shape: {stack.shape}, dtype:{stack.dtype} as TIFF file: '{tiff_file_path}', with compression: {clevel}")
                            tiff_save(tiff_file_path, stack, compress=clevel)
                            aprint(f"Done writing time point: {tp} !")
                        else:
                            aprint(f"File for time point (or z slice): {tp} already exists.")
            except Exception as error:
                aprint(error)
                aprint(f"Error occurred while processing time point {tp} !")
                import traceback
                traceback.print_exc()

                if stop_at_exception:
                    raise error

        if workers > 1:
            Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp) for tp in range(0, arrays[0].shape[0]))
        else:
            for tp in range(0, arrays[0].shape[0]):
                process(tp)

    else:

        for channel, array in zip(selected_channels, arrays):
            if len(selected_channels) > 1:
                tiff_file_path = f"{dest_path}_{channel}.tiff"
            else:
                tiff_file_path = f"{dest_path}.tiff"

            if not overwrite and os.path.exists(tiff_file_path):
                aprint(f"File {tiff_file_path} already exists! Set option -w to overwrite.")
                return

            with asection(f"Saving array ({array.shape}, {array.dtype}) for channel {channel} into TIFF file at: {tiff_file_path}:"):

                shape = array.shape

                if project is not False and type(project) == int:
                    shape = list(shape)
                    shape.pop(1+project)
                    shape = tuple(shape)

                memmap_image = memmap(tiff_file_path, shape=shape, dtype=array.dtype, bigtiff=True, imagej=True)

                def process(tp):
                    aprint(f"Processing time point {tp}")
                    stack = array[tp].compute()

                    if project is not False and type(project) == int:
                        # project is the axis for projection, but here we are not considering the T dimension anymore...
                        aprint(f"Projecting along axis {project}")
                        stack = stack.max(axis=project)

                    memmap_image[tp] = stack

                if workers > 1:
                    Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp) for tp in range(0, array.shape[0]))
                else:
                    for tp in range(0, array.shape[0]):
                        process(tp)

                memmap_image.flush()
                del memmap_image

## NOTES: color coded max projection:
# > data = numpy.random.randint(0, 255, (256, 256, 3), 'uint8')
# >>> imwrite('temp.tif', data, photometric='color')
#
# https://colorcet.holoviz.org/user_guide/index.html
# https://github.com/MMesch/cmap_builder
#
