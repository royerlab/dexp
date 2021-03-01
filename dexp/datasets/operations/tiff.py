import os
from os.path import join
from typing import Sequence

from arbol.arbol import aprint, asection
from joblib import Parallel, delayed
from tifffile import memmap

from dexp.datasets.base_dataset import BaseDataset
from dexp.io.io import tiff_save


def dataset_tiff(dataset: BaseDataset,
                 output_path: str,
                 channels: Sequence[str],
                 slicing,
                 overwrite: bool,
                 project: bool,
                 one_file_per_first_dim: bool,
                 clevel: int,
                 workers: int,
                 workersbackend: str):
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

    if workers == -1:
        workers = os.cpu_count() // 2
    aprint(f"Number of workers: {workers}")

    if one_file_per_first_dim:
        aprint(f"Saving one TIFF file for each tp (or Z if already sliced) to: {output_path}.")

        os.makedirs(output_path, exist_ok=True)

        def process(tp):
            with asection(f'Saving time point {tp}: '):
                for channel, array in zip(selected_channels, arrays):
                    tiff_file_path = join(output_path, f"file{tp}_{channel}.tiff")
                    if overwrite or not os.path.exists(tiff_file_path):
                        stack = array[tp].compute()
                        print(f"Writing time point: {tp} of shape: {stack.shape}, dtype:{stack.dtype} as TIFF file: '{tiff_file_path}', with compression: {clevel}")
                        tiff_save(tiff_file_path, stack, compress=clevel)
                        print(f"Done writing time point: {tp} !")
                    else:
                        print(f"File for time point (or z slice): {tp} already exists.")

        if workers > 1:
            Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp) for tp in range(0, arrays[0].shape[0]))
        else:
            for tp in range(0, arrays[0].shape[0]):
                process(tp)

    else:

        for channel, array in zip(selected_channels, arrays):
            if len(selected_channels) > 1:
                tiff_file_path = f"{output_path}_{channel}.tiff"
            else:
                tiff_file_path = f"{output_path}.tiff"

            if not overwrite and os.path.exists(tiff_file_path):
                aprint(f"File {tiff_file_path} already exists! Set option -w to overwrite.")
                return

            with asection(f"Saving array ({array.shape}, {array.dtype}) for channel {channel} into TIFF file at: {tiff_file_path}:"):
                memmap_image = memmap(tiff_file_path, shape=array.shape, dtype=array.dtype, bigtiff=True, imagej=True)

                def process(tp):
                    aprint(f"Processing time point {tp}")
                    stack = array[tp].compute()
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
