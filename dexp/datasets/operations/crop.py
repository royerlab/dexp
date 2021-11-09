from typing import Sequence, Optional, Tuple

import zarr
import numpy as np
from numpy.typing import ArrayLike
from arbol.arbol import aprint, asection

from dexp.datasets.base_dataset import BaseDataset
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.backends.backend import Backend
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.utils.misc import compute_num_workers
from scipy import ndimage as ndi


from joblib import Parallel, delayed


def _estimate_crop(array: ArrayLike, quantile: float = 0.99) -> Sequence[Tuple[int]]:
    window_size = 31
    step = 4
    shape = array.shape
    with BestBackend():
        xp = Backend.get_xp_module()
        array = Backend.to_backend(array[::step,::step,::step], dtype=xp.float16)
        array = xp.clip(array - xp.mean(array), 0, None)  # removing background noise
        kernel = xp.ones((window_size, window_size, window_size)) / (window_size ** 3)
        kernel = kernel.astype(xp.float16)
        array = fft_convolve(array, kernel, in_place=True)
        lower = xp.quantile(array, quantile)
        aprint('Estimated lower threshold', lower)
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
        raise RuntimeError('Could not detect any objects')

    return tuple((s.start * step, min(s.stop * step, d)) # fixing possible mismatch due to step
        for s, d in zip(largest_slice, shape))


def compute_crop_slicing(array: zarr.Array, time_points: Sequence[int], quantile: float) -> Sequence[slice]:
    # N x D x 2
    ranges = np.array(tuple(_estimate_crop(array[t], quantile) for t in time_points))
    lower = np.min(ranges[..., 0], axis=0)
    upper = np.max(ranges[..., 1], axis=0)
    return tuple(slice(int(l), int(u)) for l, u in zip(lower, upper))


def dataset_crop(dataset: BaseDataset,
                 dest_path: str,
                 reference_channel: str,
                 channels: Sequence[str],
                 quantile: float,
                 store: str = 'dir',
                 chunks: Optional[Sequence[int]] = None,
                 compression: str = 'zstd',
                 compression_level: int = 3,
                 overwrite: bool = False,
                 workers: int = 1,
                 check: bool = True,
                 stop_at_exception: bool = True,
                 ):

    # Create destination dataset:
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(dest_path, mode, store, parent=dataset)

    with asection("Estimating region of interest"):
        nb_time_pts = dataset.nb_timepoints(reference_channel)
        slicing = compute_crop_slicing(dataset.get_array(reference_channel), [0, nb_time_pts // 2, nb_time_pts-1], quantile)
        aprint('Estimated slicing of', slicing)
        volume_shape = tuple(s.stop - s.start for s in slicing)
        translation = {
            k: s.start for k, s in zip(('tz', 'ty', 'tx'), slicing)
        }
        dest_dataset.append_metadata(translation)

    # Process each channel:
    for channel in dataset._selected_channels(channels):
        with asection(f"Cropping channel {channel}:"):
            array = dataset.get_array(channel)

            dtype = array.dtype
            dest_dataset.add_channel(name=channel,
                                     shape=(len(array),) + volume_shape,
                                     dtype=dtype,
                                     chunks=chunks,
                                     codec=compression,
                                     clevel=compression_level)

            def process(tp):
                try:
                    aprint(f"Processing time point: {tp} ...")
                    tp_array = array[tp][slicing]
                    dest_dataset.write_stack(channel=channel,
                                             time_point=tp,
                                             stack_array=tp_array)
                except Exception as error:
                    aprint(error)
                    aprint(f"Error occurred while copying time point {tp} !")
                    import traceback
                    traceback.print_exc()
                    if stop_at_exception:
                        raise error

            if workers == 1:
                for i in range(len(array)):
                    process(i)
            else:
                n_jobs = compute_num_workers(workers, len(array))

                parallel = Parallel(n_jobs=n_jobs)
                parallel(delayed(process)(i) for i in range(len(array)))

    # Dataset info:
    aprint(dest_dataset.info())

    # Check dataset integrity:
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
