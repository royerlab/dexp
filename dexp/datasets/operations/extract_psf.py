import warnings
from typing import Callable, Sequence

import dask
import numpy as np
from arbol import aprint, asection
from toolz import curry

from dexp.datasets.stack_iterator import StackIterator
from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.remove_beads import BeadsRemover
from dexp.utils.backends.best_backend import BestBackend
from dexp.utils.dask import get_dask_client


@curry
def _process(
    time_point: int,
    stacks: StackIterator,
    create_beads_remover: Callable[..., BeadsRemover],
) -> np.ndarray:

    with BestBackend() as bkd:
        beads_remover = create_beads_remover()
        with asection(f"Removing beads from time point {time_point}."):
            return beads_remover.detect_beads(bkd.to_backend(stacks[time_point]))


def dataset_extract_psf(
    input_dataset: ZDataset,
    dest_path: str,
    channels: Sequence[str],
    peak_threshold: int,
    similarity_threshold: float,
    psf_size: int,
    devices: Sequence[int],
    verbose: bool = True,
) -> None:
    """
    Computes PSF from beads.
    Additional information at dexp.processing.remove_beads.beadremover documentation.
    """
    warnings.warn("This command is subpar, it should be improved.")
    dest_path = dest_path.split(".")[0]

    client = get_dask_client(devices)
    aprint("Dask client", client)

    def create_beads_remover() -> BeadsRemover:
        return BeadsRemover(
            peak_threshold=peak_threshold, similarity_threshold=similarity_threshold, psf_size=psf_size, verbose=verbose
        )

    for ch in channels:
        stacks = input_dataset[ch]
        lazy_computations = [_process(t, stacks, create_beads_remover) for t in range(len(stacks))]

        psf = np.stack(dask.compute(*lazy_computations)).mean(axis=0)
        np.save(dest_path + ch + ".npy", psf)

    client.close()
