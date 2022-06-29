from typing import Optional

import numpy as np
from arbol import aprint, asection
from joblib import Parallel, delayed
from toolz import curry

from dexp.datasets import CCDataset, ZDataset
from dexp.utils.misc import compute_num_workers


@curry
def _write(in_ds: CCDataset, out_ds: ZDataset, channel: str, indices: np.ndarray, index: int):
    aprint(f"Writing time point {index} of channel {channel}")
    stack = in_ds.get_stack(channel, indices[index])
    out_ds.write_stack(channel, index, stack)


def dataset_fromraw(
    in_dataset: CCDataset,
    out_dataset: ZDataset,
    channel_prefix: Optional[str],
    workers: int,
) -> None:

    if channel_prefix is not None:
        channels = list(filter(lambda x: x.startswith(channel_prefix), in_dataset.channels()))
    else:
        channels = in_dataset.channels()

    aprint(f"Selected channels {channels}")

    argmin = channels[0]
    for ch in channels:
        if in_dataset.nb_timepoints(ch) < in_dataset.nb_timepoints(argmin):
            argmin = ch

    min_secs = in_dataset.time_sec(argmin)
    min_time_pts = in_dataset.nb_timepoints(argmin)

    ch_to_mask = {ch: np.ones(in_dataset.nb_timepoints(ch), dtype=bool) for ch in channels}

    aprint(f"Min. time points is {min_time_pts} at channel {argmin}.")

    for ch in channels:
        ch_secs = in_dataset.time_sec(ch)

        with asection(f"Channel {ch} has {len(ch_secs)} time points."):
            while ch_to_mask[ch].sum() > min_time_pts:
                min_shift = ch_to_mask[ch].sum() - 1
                min_cost = np.abs(ch_secs[: len(min_secs)] - min_secs).sum()  # this should be the current cost
                for i in range(min_time_pts - 1):
                    mask = ch_to_mask[ch].copy()
                    if not mask[i]:
                        continue
                    mask[i] = False
                    cost = np.abs(ch_secs[mask][: len(min_secs)] - min_secs).sum()
                    if cost < min_cost:
                        min_cost = cost
                        min_shift = i

                aprint(f"Channel {ch} min. shift at {min_shift}.")
                ch_to_mask[ch][min_shift] = False

        aprint(f"Channel {ch} min. shifts: {np.where(np.logical_not(ch_to_mask[ch]))[0]}")

    for ch in channels:
        out_dataset.add_channel(ch, in_dataset.shape(argmin), in_dataset.dtype(ch))
        write_fun = _write(in_dataset, out_dataset, ch, np.where(ch_to_mask[ch])[0])

        if workers == 1:
            for i in range(min_time_pts):
                write_fun(i)
        else:
            n_jobs = compute_num_workers(workers, min_time_pts)
            parallel = Parallel(n_jobs=n_jobs)
            parallel(delayed(write_fun)(i) for i in range(min_time_pts))

    aprint(out_dataset.info())
    out_dataset.check_integrity()
    out_dataset.close()
