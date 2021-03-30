import os
from typing import Sequence

import numpy
from arbol.arbol import aprint, asection

from dexp.datasets.base_dataset import BaseDataset


def dataset_copy(dataset: BaseDataset,
                 path: str,
                 channels: Sequence[str],
                 slicing,
                 store: str,
                 compression: str,
                 compression_level: int,
                 overwrite: bool,
                 zerolevel: int,
                 workers: int,
                 workersbackend: str,
                 check: bool):
    # Create destination dataset:
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(path, mode, store)

    # Process each channel:
    for channel in dataset._selected_channels(channels):

        with asection(f"Copying channel {channel}:"):
            array = dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

            if slicing is not None:
                array = array[slicing]

            shape = array.shape
            dtype = array.dtype
            chunks = ZDataset._default_chunks

            dest_dataset.add_channel(name=channel,
                                     shape=shape,
                                     dtype=dtype,
                                     chunks=chunks,
                                     codec=compression,
                                     clevel=compression_level)

            def process(tp):
                try:
                    aprint(f"Processing time point: {tp} ...")
                    tp_array = array[tp].compute()
                    if zerolevel != 0:
                        tp_array = numpy.clip(tp_array, a_min=zerolevel)
                        tp_array -= zerolevel
                    dest_dataset.write_stack(channel=channel,
                                             time_point=tp,
                                             stack_array=tp_array)
                except Exception as error:
                    aprint(error)
                    aprint(f"Error occurred while copying time point {tp} !")

            from joblib import Parallel, delayed

            if workers == -1:
                workers = os.cpu_count() // 2

            aprint(f"Number of workers: {workers}")
            Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp) for tp in range(0, shape[0]))

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
