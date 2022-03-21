from typing import Sequence, Tuple

from arbol import asection
from toolz import curry

from dexp.datasets import BaseDataset, ZDataset
from dexp.processing.denoising import calibrate_denoise_butterworth, denoise_butterworth
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.backends import CupyBackend


def dataset_denoise(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    tilesize: Tuple[int],
):

    for ch in channels:
        stacks = input_dataset[ch]
        output_dataset.add_channel(ch, stacks.shape, dtype=input_dataset.dtype(ch))

        with CupyBackend() as bkd:
            for t, stack in enumerate(stacks):
                with asection(f"Calibrating and denoising time point {t}"):
                    stack = bkd.to_backend(stack)
                    _, best_params = calibrate_denoise_butterworth(stack)
                    denoise_fun = curry(denoise_butterworth, **best_params)
                    with asection("Denoising"):
                        denoised = scatter_gather_i2i(function=denoise_fun, image=stack, tiles=tilesize, margins=32)
                    output_dataset.write_stack(ch, t, bkd.to_numpy(denoised))
