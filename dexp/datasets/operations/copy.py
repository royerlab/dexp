from typing import Sequence, Optional

import numpy
from arbol.arbol import aprint, asection

from dexp.datasets.base_dataset import BaseDataset
from dexp.utils.slicing import slice_from_shape
from dexp.utils.misc import compute_num_workers

from joblib import Parallel, delayed


def dataset_copy(dataset: BaseDataset,
                 dest_path: str,
                 channels: Sequence[str],
                 slicing,
                 store: str = 'dir',
                 chunks: Optional[Sequence[int]] = None,
                 compression: str = 'zstd',
                 compression_level: int = 3,
                 overwrite: bool = False,
                 zerolevel: int = 0,
                 workers: int = 1,
                 workersbackend: Optional[str] = None,
                 check: bool = True,
                 stop_at_exception: bool = True,
                 ):

    # Create destination dataset:
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(dest_path, mode, store, parent=dataset)

    # Process each channel:
    for channel in dataset._selected_channels(channels):
        with asection(f"Copying channel {channel}:"):
            array = dataset.get_array(channel)

            aprint(f"Slicing with: {slicing}")
            out_shape, volume_slicing, time_points = slice_from_shape(array.shape, slicing)

            dtype = array.dtype
            dest_dataset.add_channel(name=channel,
                                     shape=out_shape,
                                     dtype=dtype,
                                     chunks=chunks,
                                     codec=compression,
                                     clevel=compression_level)

            def process(i):
                tp = time_points[i]
                try:
                    aprint(f"Processing time point: {i} ...")
                    tp_array = array[tp][volume_slicing]
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

            if workers == 1:
                for i in range(len(time_points)):
                    process(i)
            else:
                n_jobs = compute_num_workers(workers, len(time_points))

                parallel = Parallel(n_jobs=n_jobs, backend=workersbackend)
                parallel(delayed(process)(i) for i in range(len(time_points)))

    # Dataset info:
    aprint(dest_dataset.info())

    # Check dataset integrity:
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
