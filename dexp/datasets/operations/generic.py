from typing import Callable, Optional, Sequence, Tuple

import dask
from arbol import aprint, asection
from toolz import curry

from dexp.datasets import BaseDataset, ZDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.backends import CupyBackend
from dexp.utils.dask import get_dask_client


@dask.delayed
@curry
def _process(
    time_point: int,
    stacks: StackIterator,
    out_dataset: ZDataset,
    channel: str,
    func: Callable,
) -> None:

    with CupyBackend() as bkd:
        with asection(f"Applying {func.__name__} for channel {channel} at time point {time_point}"):
            stack = bkd.to_backend(stacks[time_point])
            stack = func(stack)
            out_dataset.write_stack(channel, time_point, bkd.to_numpy(stack))


def dataset_generic(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    func: Callable,
    tilesize: Optional[Tuple[int]],
    devices: Sequence[int],
) -> None:
    lazy_computations = []

    if tilesize is not None:
        func = curry(scatter_gather_i2i, function=func, tiles=tilesize, margins=32)

    for ch in channels:
        stacks = input_dataset[ch]
        output_dataset.add_channel(ch, stacks.shape, dtype=input_dataset.dtype(ch))

        process = _process(stacks=stacks, out_dataset=output_dataset, channel=ch, func=func)

        # Stores functions to be computed
        lazy_computations += [process(time_point=t) for t in range(len(stacks))]

    client = get_dask_client(devices)
    aprint("Dask client", client)

    # Compute everything
    dask.compute(*lazy_computations)

    output_dataset.check_integrity()
