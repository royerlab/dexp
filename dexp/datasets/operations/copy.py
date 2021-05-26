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
                 stop_at_exception: bool = True,
                 ):

    # Create destination dataset:
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(dest_path, mode, store)

    # Process each channel:
    for channel in dataset._selected_channels(channels):
        with asection(f"Copying channel {channel}:"):
            array = dataset.get_array(channel)

            total_time_points = shape[0]
            time_points = list(range(total_time_points))
            if slicing is not None:
                aprint(f"Slicing with: {slicing}")
                if isinstance(slicing, tuple):
                    time_points = time_points[slicing[0]]
                    slicing = slicing[1:]
                else:  # slicing only over time
                    time_points = time_points[slicing]
                    slicing = ...
            else:
                slicing = ...

            shape = array[0][slicing].shape
            dtype = array.dtype
            if chunks is None:
                chunks = ZDataset._default_chunks

            dest_dataset.add_channel(name=channel,
                                     shape=shape,
                                     dtype=dtype,
                                     chunks=chunks,
                                     codec=compression,
                                     clevel=compression_level)

            def process(i):
                tp = time_points[i]
                try:
                    aprint(f"Processing time point: {i} ...")
                    tp_array = array[tp][slicing]
                    if zerolevel != 0:
                        tp_array = numpy.array(tp_array)
                        tp_array = numpy.clip(tp_array, a_min=zerolevel, a_max=None, out=tp_array)
                        tp_array -= zerolevel
                    dest_dataset.write_stack(channel=channel,
                                             time_point=i,
                                             stack_array=tp_array)
                except Exception as error:
                    aprint(error)
                    aprint(f"Error occurred while copying time point {i} !")
                    import traceback
                    traceback.print_exc()
                    if stop_at_exception:
                        raise error

            for i in range(len(time_points)):
                process(i)

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    dest_dataset.set_cli_history(parent=dataset if isinstance(dataset, ZDataset) else None)
    # close destination dataset:
    dest_dataset.close()
