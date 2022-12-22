from typing import Callable, Dict, Optional, Sequence

import dask
import numpy as np
from arbol import aprint, asection, section
from toolz import curry, reduce

from dexp.datasets import BaseDataset, ZDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.processing.crop.background import foreground_mask
from dexp.utils.backends.cupy_backend import CupyBackend
from dexp.utils.dask import get_dask_client


@dask.delayed
@curry
def _process(
    time_point: int,
    time_scale: Dict[str, int],
    stacks: Dict[str, StackIterator],
    out_dataset: ZDataset,
    reference_channel: Optional[str],
    merge_channels: bool,
    foreground_mask_func: Callable,
) -> None:

    foreground_mask_func = section("Detecting foreground.")(foreground_mask_func)

    with CupyBackend() as bkd:

        # selects stacks while scaling time points
        time_pts = {ch: int(round(time_point / s)) for ch, s in time_scale.items()}
        stacks = {ch: bkd.to_backend(stacks[ch][time_pts[ch]]) for ch in stacks.keys()}

        with asection(f"Removing background of channels {time_pts.keys()} at time points {time_pts.values()}."):
            # computes reference mask (or not)
            if merge_channels:
                reference = foreground_mask_func(reduce(np.add, stacks.values()))
            elif reference_channel is not None:
                reference = foreground_mask_func(stacks[reference_channel])
            else:
                reference = None

            # removes background
            for ch in stacks.keys():
                stack = stacks[ch]
                mask = foreground_mask_func(stack) if reference is None else reference
                stack[np.logical_not(mask)] = 0
                out_dataset.write_stack(ch, time_pts[ch], bkd.to_numpy(stack))


def dataset_background(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    reference_channel: Optional[str],
    merge_channels: bool,
    devices: Sequence[int],
    **kwargs,
) -> None:
    lazy_computations = []

    if reference_channel is not None and merge_channels:
        raise ValueError("`reference_channel` cannot be supplied with `merge_channels` option.")

    arrays = {ch: input_dataset[ch] for ch in channels}
    max_t = max(len(arr) for arr in arrays.values())
    time_scale = {ch: input_dataset.get_resolution(ch)[0] for ch in channels}
    process = _process(
        time_scale=time_scale,
        stacks=arrays,
        out_dataset=output_dataset,
        reference_channel=reference_channel,
        merge_channels=merge_channels,
        foreground_mask_func=curry(foreground_mask, **kwargs),
    )

    for ch in channels:
        output_dataset.add_channel(ch, input_dataset.shape(ch), input_dataset.dtype(ch))

    lazy_computations = [process(time_point=t) for t in range(max_t)]

    client = get_dask_client(devices)
    aprint("Dask client", client)

    # Compute everything
    dask.compute(*lazy_computations)

    output_dataset.check_integrity()
