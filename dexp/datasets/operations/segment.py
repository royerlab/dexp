from typing import Sequence

import dask
import numpy as np
import pandas as pd
from arbol import aprint, asection
from skimage.measure import regionprops_table
from toolz import curry

from dexp.datasets import BaseDataset, ZDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.processing.morphology import area_white_top_hat
from dexp.processing.segmentation import roi_watershed_from_minima
from dexp.utils.backends import CupyBackend
from dexp.utils.dask import get_dask_client


def intensity_sum(mask: np.ndarray, intensities: np.ndarray) -> float:
    """Sums the intensity inside the mask"""
    return intensities[mask].sum(axis=0, dtype=float)


@curry
def _process(
    detection_stacks: Sequence[StackIterator],
    feature_stacks: Sequence[StackIterator],
    out_dataset: ZDataset,
    out_channel: str,
    time_point: int,
    z_scale: float,
    area_threshold: float,
    minimum_area: float,
    h_minima: float,
    compactness: float,
    gamma: float,
    use_edt: bool,
) -> pd.DataFrame:
    from cucim.skimage import morphology as morph
    from cucim.skimage.filters import threshold_otsu
    from edt import edt

    with CupyBackend() as bkd:
        with asection(f"Segmenting time point {time_point}:"):
            xp = bkd.get_xp_module()

            detection = xp.zeros(detection_stacks[0].shape[1:], dtype=bool)

            if use_edt:
                basins = None
            else:
                basins = xp.zeros(detection_stacks[0].shape[1:], np.float32)

            # detects each channel individually and merge then into a single image
            for i, stacks in enumerate(detection_stacks):
                stack = bkd.to_backend(stacks[time_point])

                with asection(f"Morphological filtering of channel {i} ..."):
                    filtered = morph.closing(stack, morph.ball(np.sqrt(2)))
                    if not np.isclose(gamma, 1.0):
                        # the results are the same when applying the power before and after morph closing
                        filtered = np.power(filtered.astype(np.float32), gamma)

                    if basins is not None:  # not using EDT
                        basins += filtered / np.quantile(filtered, 0.999)

                    wth = area_white_top_hat(filtered, area_threshold, sampling=4, axis=0)

                    del filtered

                with asection(f"Detecting cells of channel {i} ..."):
                    ch_detection = wth > threshold_otsu(wth)
                    del wth

                    # removing small white/black elements
                    ch_detection = morph.binary_opening(ch_detection, morph.ball(2))
                    ch_detection = morph.binary_closing(ch_detection, morph.ball(2))

                    # FIXME: it needs to be executed after labeling
                    # ch_detection = morph.remove_small_objects(ch_detection, min_size=minimum_area)

                    detection |= ch_detection
                    del ch_detection

            count = detection.sum()
            aprint(f"Number of detected cell-pixels {count} proportion {detection.sum() / detection.size}.")

            if basins is None:  # using EDT
                with asection("Computing EDT for watershed basins ..."):
                    basins = bkd.to_backend(edt(bkd.to_numpy(detection), anisotropy=(z_scale, 1, 1)))

            basins = basins.max() - basins
            basins = bkd.to_numpy(basins)
            detection = bkd.to_numpy(detection)
            bkd.clear_memory_pool()

            with asection("Segmenting ..."):
                labels = roi_watershed_from_minima(
                    image=basins,
                    mask=detection,
                    H_minima=h_minima,
                    compactness=0 if use_edt else compactness,
                    scales=None if use_edt else (z_scale, 1, 1),
                )
                del basins, detection
                labels = labels.astype(np.int32)

            out_dataset.write_stack(out_channel, time_point, labels)

        feature_image = np.stack(
            [stack[time_point] for stack in feature_stacks],
            axis=-1,
        )

        df = pd.DataFrame(
            regionprops_table(
                label_image=labels,
                intensity_image=feature_image,
                cache=False,
                properties=["label", "area", "bbox", "centroid", "intensity_max", "intensity_min", "intensity_mean"],
                extra_properties=(intensity_sum,),
            )
        )
        df["time_point"] = time_point

    aprint(df.describe())

    return df


def _validate_stacks_shapes(stacks: Sequence[StackIterator], channels: Sequence[str]) -> None:
    if any(s.shape != stacks[0].shape for s in stacks):
        raise ValueError(f"All stacks must have the same shape, found {[s.shape for s in stacks]} for {channels}")


def dataset_segment(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    detection_channels: Sequence[str],
    features_channels: Sequence[str],
    out_channel: str,
    devices: Sequence[int],
    z_scale: float,
    area_threshold: float,
    minimum_area: float,
    h_minima: float,
    compactness: float,
    use_edt: bool,
    gamma: float,
) -> None:

    detection_stacks = [input_dataset[ch] for ch in detection_channels]
    feature_stacks = [input_dataset[ch] for ch in features_channels]

    _validate_stacks_shapes(detection_stacks + feature_stacks, list(detection_channels) + list(features_channels))

    output_dataset.add_channel(out_channel, detection_stacks[0].shape, dtype=np.int32, value=0)

    n_time_pts = len(detection_stacks[0])

    process = dask.delayed(
        _process(
            detection_stacks=detection_stacks,
            feature_stacks=feature_stacks,
            out_dataset=output_dataset,
            z_scale=z_scale,
            area_threshold=area_threshold,
            minimum_area=minimum_area,
            h_minima=h_minima,
            compactness=compactness,
            gamma=gamma,
            use_edt=use_edt,
            out_channel=out_channel,
        )
    )

    lazy_computations = [process(time_point=t) for t in range(n_time_pts)]

    client = get_dask_client(devices)
    aprint("Dask client", client)

    df_path = output_dataset.path.replace(".zarr", ".csv")
    df = pd.concat(dask.compute(*lazy_computations))
    df.to_csv(df_path, index=False)

    output_dataset.append_metadata({"features": features_channels})

    output_dataset.check_integrity()
