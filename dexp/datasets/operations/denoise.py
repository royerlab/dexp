from typing import Sequence

from arbol import asection

from dexp.datasets import BaseDataset, ZDataset
from dexp.processing.denoising import calibrate_denoise_butterworth, denoise_butterworth
from dexp.utils.backends import CupyBackend


def dataset_denoise(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
):

    for ch in channels:
        stacks = input_dataset[ch]
        output_dataset.add_channel(ch, stacks.shape, dtype=input_dataset.dtype(ch))

        with CupyBackend() as bkd:
            for t, stack in enumerate(stacks):
                with asection(f"Calibrating and denoising time point {t}"):
                    stack = bkd.to_backend(stack)
                    _, best_params = calibrate_denoise_butterworth(stack)
                    with asection("Denoising"):
                        denoised = denoise_butterworth(stack, **best_params)
                    output_dataset.write_stack(ch, t, bkd.to_numpy(denoised))
