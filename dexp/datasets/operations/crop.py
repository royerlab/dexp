from typing import Sequence, Tuple

import numpy as np
import zarr
from arbol.arbol import aprint, asection
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy import ndimage as ndi
from toolz import curry

from dexp.datasets import BaseDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.utils.backends import Backend, BestBackend
from dexp.utils.misc import compute_num_workers


def _estimate_crop(array: ArrayLike, quantile: float) -> Sequence[Tuple[int]]:
    window_size = 15
    step = 4
    shape = array.shape
    with BestBackend():
        xp = Backend.get_xp_module()
        array = Backend.to_backend(array[::step, ::step, ::step], dtype=xp.float16)
        array = xp.clip(array - xp.mean(array), 0, None)  # removing background noise
        kernel = xp.ones((window_size, window_size, window_size)) / (window_size**3)
        kernel = kernel.astype(xp.float16)
        array = fft_convolve(array, kernel, in_place=True)
        lower = xp.quantile(array, quantile)
        aprint("Estimated lower threshold", lower)
        array = array > lower
        array, _ = ndi.label(Backend.to_numpy(array))
        slices = ndi.find_objects(array)

    largest_slice = None
    largest_size = 0
    for slicing in slices:
        size = np.prod(tuple((s.stop - s.start) for s in slicing))
        if size > largest_size:
            largest_slice = slicing
            largest_size = size

    if largest_slice is None:
        raise RuntimeError("Could not detect any objects")

    return tuple(
        (s.start * step, min(s.stop * step, d))  # fixing possible mismatch due to step
        for s, d in zip(largest_slice, shape)
    )


def compute_crop_slicing(array: zarr.Array, time_points: Sequence[int], quantile: float) -> Sequence[slice]:
    # N x D x 2
    ranges = np.array(tuple(_estimate_crop(array[t], quantile) for t in time_points))
    lower = np.min(ranges[..., 0], axis=0)
    upper = np.max(ranges[..., 1], axis=0)
    return tuple(slice(int(l), int(u)) for l, u in zip(lower, upper))


@curry
def _process(i: int, array: StackIterator, output_dataset: ZDataset, channel: str):
    try:
        aprint(f"Processing time point: {i} ...")
        output_dataset.write_stack(channel=channel, time_point=i, stack_array=np.asarray(array[i]))
    except Exception as error:
        aprint(error)
        aprint(f"Error occurred while copying time point {i} !")
        import traceback

        traceback.print_exc()
        raise error


def dataset_crop(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    reference_channel: str,
    channels: Sequence[str],
    quantile: float,
    workers: int,
):
    with asection("Estimating region of interest"):
        nb_time_pts = input_dataset.nb_timepoints(reference_channel)
        slicing = compute_crop_slicing(
            input_dataset.get_array(reference_channel), [0, nb_time_pts // 2, nb_time_pts - 1], quantile
        )
        aprint("Estimated slicing of", slicing)
        translation = {k: s.start for k, s in zip(("tz", "ty", "tx"), slicing)}
        output_dataset.append_metadata(translation)
        input_dataset.set_slicing(slicing=(slice(None, None),) + slicing)

    # Process each channel:
    for channel in input_dataset._selected_channels(channels):
        with asection(f"Cropping channel {channel}:"):
            array = input_dataset[channel]
            output_dataset.add_channel(name=channel, shape=array.shape, dtype=array.dtype)
            process = _process(array=array, output_dataset=output_dataset, channel=channel)

            if workers == 1:
                for i in range(len(array)):
                    process(i)
            else:
                n_jobs = compute_num_workers(workers, len(array))
                parallel = Parallel(n_jobs=n_jobs)
                parallel(delayed(process)(i) for i in range(len(array)))

    # Dataset info:
    aprint(output_dataset.info())

    # Check dataset integrity:
    output_dataset.check_integrity()

    # close destination dataset:
    output_dataset.close()
