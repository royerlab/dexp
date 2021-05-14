import os
from typing import Sequence, Optional

import numpy
from arbol.arbol import aprint, asection

from dexp.datasets.base_dataset import BaseDataset


def dataset_copy(dataset: BaseDataset,
                 dest_path: str,
                 channels: Sequence[str],
                 slicing,
                 store: str,
                 chunks: Optional[Sequence[int]],
                 compression: str,
                 compression_level: int,
                 overwrite: bool,
                 zerolevel: int,
                 workers: int,
                 workersbackend: str,
                 check: bool,
                 stop_at_exception: bool = True):

    # Create destination dataset:
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(dest_path, mode, store)

    # Process each channel:
    for channel in dataset._selected_channels(channels):

        with asection(f"Copying channel {channel}:"):
            array = dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

            if slicing is not None:
                array = array[slicing]

            shape = array.shape
            dtype = array.dtype
            if chunks is None:
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
                        tp_array = numpy.array(tp_array)
                        tp_array = numpy.clip(tp_array, a_min=zerolevel, a_max=None, out=tp_array)
                        tp_array -= zerolevel
                    dest_dataset.write_stack(channel=channel,
                                             time_point=tp,
                                             stack_array=tp_array)
                except Exception as error:
                    aprint(error)
                    aprint(f"Error occurred while copying time point {tp} !")
                    import traceback
                    traceback.print_exc()
                    if stop_at_exception:
                        raise error

            from joblib import Parallel, delayed

            if workers == -1:
                workers = max(1, os.cpu_count() // abs(workers))

            aprint(f"Number of workers: {workers}")
            Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp) for tp in range(0, shape[0]))

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    dest_dataset.set_cli_history(parent=dataset if isinstance(dataset, ZDataset) else None)
    # close destination dataset:
    dest_dataset.close()
