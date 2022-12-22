import os
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from arbol.arbol import aprint, asection
from joblib import Parallel, delayed
from tifffile import memmap
from toolz import curry

from dexp.datasets import BaseDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.io.io import tiff_save


@curry
def _save_multi_file(
    tp: int, stacks: StackIterator, dest_path: Path, channel: str, clevel: int, overwrite: bool, project: bool
) -> None:
    with asection(f"Saving time point {tp}: "):
        tiff_file_path = dest_path / f"file{tp}_{channel}.tiff"
        if overwrite or not tiff_file_path.exists():
            stack = np.asarray(stacks[tp])

            if type(project) == int:
                # project is the axis for projection, but here we are not considering
                # the T dimension anymore...
                aprint(f"Projecting along axis {project}")
                stack = stack.max(axis=project)

            aprint(
                f"Writing time point: {tp} of shape: {stack.shape}, dtype:{stack.dtype} "
                + f"as TIFF file: '{tiff_file_path}', with compression: {clevel}"
            )
            tiff_save(tiff_file_path, stack, clevel=clevel)
            aprint(f"Done writing time point: {tp} !")
        else:
            aprint(f"File for time point (or z slice): {tp} already exists.")


@curry
def _save_single_file(tp: int, stacks: StackIterator, memmap_image: np.memmap, project: bool) -> None:
    aprint(f"Processing time point {tp}")
    stack = np.asarray(stacks[tp])

    if type(project) == int:
        # project is the axis for projection, but here we are not considering the T dimension anymore...
        aprint(f"Projecting along axis {project}")
        stack = stack.max(axis=project)

    memmap_image[tp] = stack


def dataset_tiff(
    dataset: BaseDataset,
    dest_path: Path,
    channels: Sequence[str],
    overwrite: bool = False,
    project: Union[int, bool] = False,
    one_file_per_first_dim: bool = False,
    clevel: int = 0,
    workers: int = 1,
    workersbackend: str = "",
):
    aprint(f"getting Dask arrays for channels {channels}")

    if workers == -1:
        workers = max(1, os.cpu_count() // abs(workers))

    aprint(f"Number of workers: {workers}")

    if one_file_per_first_dim:
        aprint(f"Saving one TIFF file for each tp (or Z if already sliced) to: {dest_path}.")

        dest_path.mkdir(exist_ok=True)

        process = _save_multi_file(dest_path=dest_path, clevel=clevel, overwrite=overwrite, project=project)

        for channel in channels:
            stacks = dataset[channel]
            _process = process(stacks=stacks, channel=channel)

            _workers = min(workers, len(stacks))
            if _workers > 1:
                Parallel(n_jobs=_workers, backend=workersbackend)(delayed(_process)(tp) for tp in range(len(stacks)))
            else:
                for tp in range(len(stacks)):
                    _process(tp)

    else:
        for channel in channels:
            if len(channels) > 1:
                tiff_file_path = f"{dest_path}_{channel}.tiff"

            elif not dest_path.name.endswith((".tiff", ".tif")):
                tiff_file_path = f"{dest_path}.tiff"

            tiff_file_path = Path(tiff_file_path)

            if not overwrite and tiff_file_path.exists():
                aprint(f"File {tiff_file_path} already exists! Set option -w to overwrite.")
                return

            stacks = dataset[channel]
            shape = stacks.shape

            if type(project) == int:
                shape = list(shape)
                shape.pop(1 + project)
                shape = tuple(shape)

            memmap_image = memmap(tiff_file_path, shape=shape, dtype=stacks.dtype, bigtiff=True, imagej=True)

            _process = _save_single_file(stacks=stacks, memmap_image=memmap_image, project=project)

            _workers = min(len(stacks), workers)
            if _workers > 1:
                Parallel(n_jobs=_workers, backend=workersbackend)(delayed(_process)(tp) for tp in range(len(stacks)))

            else:
                for tp in range(len(stacks)):
                    _process(tp)

            memmap_image.flush()
            del memmap_image
