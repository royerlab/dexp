from typing import Sequence

import dask
import numpy as np
import pandas as pd
from arbol import aprint, asection
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential
from toolz import curry

from dexp.datasets import BaseDataset, ZDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.processing.morphology import area_white_top_hat
from dexp.utils.backends import CupyBackend


@curry
def _process(
    stacks: StackIterator,
    out_dataset: ZDataset,
    channel: str,
    time_point: int,
    z_scale: float,
    area_threshold: float,
    minimum_area: float,
    h_minima: float,
    compactness: float,
    use_edt: bool,
) -> pd.DataFrame:
    from cucim.skimage import morphology as morph
    from cucim.skimage.filters import threshold_otsu
    from edt import edt
    from pyift.shortestpath import watershed_from_minima

    with CupyBackend() as bkd:
        with asection(f"Segmenting channel {channel} time point {time_point}:"):

            stack = bkd.to_backend(stacks[time_point])
            with asection("Morphological filtering ..."):
                filtered = morph.closing(stack, morph.ball(np.sqrt(2)))
                wth = area_white_top_hat(filtered, area_threshold, sampling=4, axis=0)

            with asection("Detecting cells ..."):
                detection = wth > threshold_otsu(wth)
                detection = morph.closing(detection, morph.ball(np.sqrt(2)))
                detection = morph.remove_small_objects(detection, min_size=minimum_area)

            if use_edt:
                basins = bkd.to_backend(edt(bkd.to_numpy(detection), anisotropy=(z_scale, 1, 1)))
            else:
                basins = filtered / np.quantile(filtered, 0.999)

            basins = basins.max() - basins
            basins = bkd.to_numpy(basins)
            detection = bkd.to_numpy(detection)

            with asection("Segmenting ..."):
                _, labels = watershed_from_minima(
                    image=basins,
                    mask=detection,
                    H_minima=h_minima,
                    compactness=0 if use_edt else compactness,
                    scales=None if use_edt else (z_scale, 1, 1),
                )
                labels = labels.astype(np.int32)

            with asection("Relabeling ..."):
                labels[labels < 0] = 0
                labels, _, _ = relabel_sequential(labels)

            out_dataset.write_stack(channel, time_point, labels)

        df = pd.DataFrame(
            regionprops_table(
                label_image=labels,
                intensity_image=bkd.to_numpy(stack),
                cache=False,
                properties=["label", "area", "bbox", "centroid", "intensity_max", "intensity_min", "intensity_mean"],
            )
        )
        df["time_point"] = time_point
        df["channel"] = channel

    return df


def dataset_segment(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    suffix: str,
    devices: Sequence[int],
    z_scale: float,
    area_threshold: float,
    minimum_area: float,
    h_minima: float,
    compactness: float,
    use_edt: bool,
) -> None:

    _segment_func = _process(
        z_scale=z_scale,
        area_threshold=area_threshold,
        minimum_area=minimum_area,
        h_minima=h_minima,
        compactness=compactness,
        use_edt=use_edt,
    )

    lazy_computations = []

    for ch in channels:
        stacks = input_dataset[ch]
        out_ch = ch + suffix
        output_dataset.add_channel(out_ch, stacks.shape, dtype=np.int32, value=0)
        process = dask.delayed(_segment_func(stacks=stacks, out_dataset=output_dataset, channel=out_ch))
        lazy_computations += [process(time_point=t) for t in range(len(stacks))]

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=devices)
    client = Client(cluster)
    aprint("Dask client", client)

    df_path = output_dataset._path.replace(".zarr", ".csv")
    df = pd.concat(dask.compute(*lazy_computations))
    df.to_csv(df_path, index=False)

    output_dataset.check_integrity()
