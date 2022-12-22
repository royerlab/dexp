from typing import Callable, Sequence, Tuple

import dask
from arbol import aprint, asection
from toolz import curry

from dexp.datasets import BaseDataset, ZDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.processing.denoising import calibrate_denoise_butterworth, denoise_butterworth
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.backends import CupyBackend
from dexp.utils.dask import get_dask_client
from dexp.utils.fft import clear_fft_plan_cache


@curry
def _process(
    stacks: StackIterator,
    out_dataset: ZDataset,
    channel: str,
    time_point: int,
    scatter_gather: Callable,
) -> None:

    with CupyBackend() as bkd:
        with asection(f"Calibrating and denoising channel {channel} time point {time_point}"):
            stack = bkd.to_backend(stacks[time_point])
            _, best_params = calibrate_denoise_butterworth(stack)
            denoise_fun = curry(denoise_butterworth, **best_params)

            with asection("Denoising"):
                denoised = scatter_gather(function=denoise_fun, image=stack)

            out_dataset.write_stack(channel, time_point, bkd.to_numpy(denoised))
            clear_fft_plan_cache()


def dataset_denoise(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    tilesize: Tuple[int],
    devices: Sequence[int],
):
    # NOTE:
    #   we might want in the future to smooth the parameters estimation over neighboring time points.
    lazy_computations = []

    for ch in channels:
        stacks = input_dataset[ch]
        output_dataset.add_channel(ch, stacks.shape, dtype=input_dataset.dtype(ch))

        # Create processing function with default parameters
        process = dask.delayed(
            _process(
                stacks=stacks,
                out_dataset=output_dataset,
                channel=ch,
                scatter_gather=curry(scatter_gather_i2i, tiles=tilesize, margins=32),
            )
        )  # using 32 because Jordao assumed it's good enough and 320 (default tile) + 64 = 384
        # has a nice prime factorization, speeding up fft computation

        # Stores functions to be computed
        lazy_computations += [process(time_point=t) for t in range(len(stacks))]

    client = get_dask_client(devices)
    aprint("Dask client", client)

    # Compute everything
    dask.compute(*lazy_computations)

    output_dataset.check_integrity()
