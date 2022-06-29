from pathlib import Path
from typing import Callable, Sequence

import dask
import imageio
import numpy as np
from arbol.arbol import aprint, asection
from dask.distributed import Client
from toolz import curry

from dexp.datasets import BaseDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.processing.color.projection import project_image
from dexp.utils.backends import BestBackend
from dexp.utils.backends.cupy_backend import is_cupy_available


@dask.delayed
@curry
def _process(time_point: int, stacks: StackIterator, outpath: Path, project_func: Callable, overwrite: bool) -> None:
    with asection(f"Rendering Frame \t: {time_point:05}"):

        filename = outpath / f"frame_{time_point:05}.png"

        if overwrite or not filename.exists():

            with asection("Loading stack..."):
                stack = np.asarray(stacks[time_point])

            with BestBackend() as bkd:
                with asection(f"Projecting image of shape: {stack.shape} "):
                    projection = bkd.to_numpy(project_func(bkd.to_backend(stack)))

                with asection(f"Saving frame {time_point} as: {filename}"):
                    imageio.imwrite(filename, projection, compress_level=1)


def dataset_projection_rendering(
    input_dataset: BaseDataset,
    output_path: Path,
    channels: Sequence[str],
    overwrite: bool,
    devices: Sequence[int],
    **kwargs,
):
    project_func = curry(project_image, **kwargs)
    lazy_computations = []

    for channel in channels:
        # Ensures that the output folder exists per channel:
        if len(channels) == 1:
            channel_output_path = output_path
        else:
            channel_output_path = output_path / channel

        channel_output_path.mkdir(exist_ok=True, parents=True)

        stacks = input_dataset[channel]

        process = _process(
            stacks=stacks,
            outpath=channel_output_path,
            project_func=project_func,
            overwrite=overwrite,
        )

        with asection(f"Channel '{channel}' shape: {stacks.shape}:"):
            aprint(input_dataset.info(channel))

        lazy_computations += [process(time_point=t) for t in range(len(stacks))]

    if len(devices) > 0 and is_cupy_available():
        from dask_cuda import LocalCUDACluster

        cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=devices)
        client = Client(cluster)
    else:
        client = Client()
    aprint("Dask client", client)

    # Compute everything
    dask.compute(*lazy_computations)

    client.close()
