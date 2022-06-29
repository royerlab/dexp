from typing import Optional, Sequence

import numpy as np
from arbol.arbol import aprint, asection
from joblib import Parallel, delayed
from toolz import curry

from dexp.datasets import BaseDataset, ZDataset
from dexp.datasets.clearcontrol_dataset import CCDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.utils.misc import compute_num_workers


@curry
def _process(
    i: int,
    in_array: StackIterator,
    output_dataset: ZDataset,
    channel: str,
    zerolevel: int,
    stop_at_exception: bool = True,
) -> None:
    try:
        aprint(f"Processing time point: {i} ...")
        tp_array = np.asarray(in_array[i])
        if zerolevel != 0:
            tp_array = np.clip(tp_array, a_min=zerolevel, a_max=None, out=tp_array)
            tp_array -= zerolevel
        output_dataset.write_stack(channel=channel, time_point=i, stack_array=tp_array)

    except Exception as error:
        aprint(error)
        aprint(f"Error occurred while copying time point {i} !")
        import traceback

        traceback.print_exc()
        if stop_at_exception:
            raise error


def dataset_copy(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    zerolevel: int = 0,
    workers: int = 1,
    workersbackend: Optional[str] = None,
):
    # Process each channel:
    for channel in channels:
        array = input_dataset[channel]
        process = _process(in_array=array, output_dataset=output_dataset, channel=channel, zerolevel=zerolevel)
        with asection(f"Copying channel {channel}:"):

            aprint(f"Slicing with: {array.slicing}")
            output_dataset.add_channel(name=channel, shape=array.shape, dtype=array.dtype)

            if workers == 1:
                for i in range(len(array)):
                    process(i)
            else:
                n_jobs = compute_num_workers(workers, len(array))
                parallel = Parallel(n_jobs=n_jobs, backend=workersbackend)
                parallel(delayed(process)(i) for i in range(len(array)))

        if isinstance(input_dataset, CCDataset):
            time = input_dataset.time_sec(channel).tolist()
            output_dataset.append_metadata({channel: dict(time=time)})

    # Dataset info:
    aprint(output_dataset.info())

    # Check dataset integrity:
    output_dataset.check_integrity()

    # close destination dataset:
    output_dataset.close()
