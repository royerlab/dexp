from typing import Callable, List, Optional, Sequence

import dask
import fasteners
from arbol.arbol import aprint, asection
from toolz import curry

from dexp.datasets import BaseDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.deskew import deskew_functions
from dexp.utils.backends import BestBackend
from dexp.utils.dask import get_dask_client
from dexp.utils.lock import create_lock


@dask.delayed
@curry
def _process(
    time_point: int,
    stacks: StackIterator,
    channel: str,
    output_dataset: ZDataset,
    lock: fasteners.InterProcessLock,
    deskew_func: Callable,
) -> None:
    with asection(f"Deskweing channel {channel} at time point {time_point}."):
        with BestBackend() as bkd:
            with asection("Loading data"):
                stack = bkd.to_backend(stacks[time_point])
            with asection("Processing"):
                stack = deskew_func(stack)
            stack = bkd.to_numpy(stack)

        with lock:
            if channel not in output_dataset:
                output_dataset.add_channel(
                    channel,
                    (len(stacks),) + stack.shape,
                    dtype=stack.dtype,
                )

        with asection("Saving deskwed array"):
            output_dataset.write_stack(channel, time_point, stack)


def dataset_deskew(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    dx: Optional[float],
    dz: Optional[float],
    angle: Optional[float],
    flips: Sequence[bool],
    camera_orientation: int,
    depth_axis: int,
    lateral_axis: int,
    mode: str,
    padding: bool,
    devices: List[int],
):
    # Default flipping:
    if flips is None:
        flips = (False,) * len(channels)

    # Metadata for deskewing:
    metadata = input_dataset.get_metadata()
    aprint(f"Dataset metadata: {metadata}")
    if dx is None and "res" in metadata:
        dx = float(metadata["res"])
    if dz is None and "dz" in metadata:
        dz = float(metadata["dz"])
    if angle is None and "angle" in metadata:
        angle = float(metadata["angle"])

    # setting up fixed parameters
    aprint(f"Deskew parameters: dx={dx}, dz={dz}, angle={angle}")
    deskew_func = curry(
        deskew_functions[mode],
        depth_axis=depth_axis,
        lateral_axis=lateral_axis,
        dx=dx,
        dz=dz,
        angle=angle,
        camera_orientation=camera_orientation,
        padding=padding,
    )

    lazy_computations = []

    # Iterate through channels
    for i, channel in enumerate(channels):
        stacks = input_dataset[channel]
        lock = create_lock(channel)
        lazy_computations += [
            _process(
                time_point=t,
                stacks=stacks,
                channel=channel,
                lock=lock,
                output_dataset=output_dataset,
                deskew_func=deskew_func(flip_depth_axis=flips[i]),
            )
            for t in range(len(stacks))
        ]

    # setting up dask compute scheduler
    client = get_dask_client(devices)

    aprint("Dask client", client)
    dask.compute(*lazy_computations)

    # shape and dtype of views to deskew:
    output_dataset.check_integrity()

    aprint(output_dataset.info())
