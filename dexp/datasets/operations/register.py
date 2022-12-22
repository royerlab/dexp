from copy import deepcopy
from typing import Dict, List, Sequence

import dask
import numpy
import numpy as np
from arbol.arbol import aprint, asection
from toolz import curry

from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.processing.multiview_lightsheet.fusion.basefusion import BaseFusion
from dexp.processing.multiview_lightsheet.fusion.simview import SimViewFusion
from dexp.processing.registration.model.model_io import model_list_to_file
from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.utils.backends import BestBackend
from dexp.utils.dask import get_dask_client


@dask.delayed
@curry
def _process(
    tp: int,
    views: Dict[str, StackIterator],
    fuse_model: BaseFusion,
    max_proj: bool,
    registration_edge_filter: bool,
) -> TranslationRegistrationModel:

    fuse_model = deepcopy(fuse_model)

    with BestBackend(exclusive=True, enable_unified_memory=True):
        with asection(f"Loading volume {tp}"):
            views_tp = {k: np.asarray(view[tp]) for k, view in views.items()}

        with asection(f"Registring volume {tp}:"):
            C0Lx, C1Lx = fuse_model.preprocess(**views_tp)
            fuse_model.compute_registration(
                C0Lx,
                C1Lx,
                mode="projection" if max_proj else "full",
                edge_filter=registration_edge_filter,
                crop_factor_along_z=0.3,
            )
            model = fuse_model.registration_model.to_numpy()

    return model


def dataset_register(
    dataset: BaseDataset,
    model_path: str,
    channels: Sequence[str],
    microscope: str,
    equalise: bool,
    zero_level: int,
    clip_too_high: int,
    fusion: str,
    fusion_bias_strength_i: float,
    dehaze_size: int,
    registration_edge_filter: bool,
    max_proj: bool,
    white_top_hat_size: float,
    white_top_hat_sampling: int,
    remove_beads: bool,
    devices: Sequence[int],
) -> None:

    views = {channel.split("-")[-1]: dataset[channel] for channel in channels}
    n_time_pts = len(list(views.values())[0])

    with asection("Views:"):
        for channel, view in views.items():
            aprint(f"View: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    if microscope == "simview":
        fuse_model = SimViewFusion(
            registration_model=None,
            equalise=equalise,
            equalisation_ratios=[None, None, None],
            zero_level=zero_level,
            clip_too_high=clip_too_high,
            fusion=fusion,
            fusion_bias_exponent=2,
            fusion_bias_strength_i=fusion_bias_strength_i,
            fusion_bias_strength_d=0.0,
            dehaze_before_fusion=True,
            dehaze_size=dehaze_size,
            dehaze_correct_max_level=True,
            dark_denoise_threshold=0,
            dark_denoise_size=0,
            butterworth_filter_cutoff=0.0,
            white_top_hat_size=white_top_hat_size,
            white_top_hat_sampling=white_top_hat_sampling,
            remove_beads=remove_beads,
            flip_camera1=True,
        )
        views = fuse_model.validate_views(views)
    else:
        raise NotImplementedError

    process = _process(
        views=views, fuse_model=fuse_model, max_proj=max_proj, registration_edge_filter=registration_edge_filter
    )

    client = get_dask_client(devices)
    aprint("Dask Client", client)

    lazy_computations = []
    for i in range(n_time_pts):
        lazy_computations.append(process(tp=i))

    models = dask.compute(*lazy_computations)

    mode_model = compute_median_translation(models)
    model_list_to_file(model_path, [mode_model])

    client.close()


def compute_median_translation(models: List[TranslationRegistrationModel]) -> TranslationRegistrationModel:
    """
    Computes the median of each axis over all registration models and
    assigns the confidence to the one with smallest distance to it.
    """
    models_shift = np.vstack([m.shift_vector for m in models if not numpy.isnan(m.confidence)])
    translation_mode = np.median(models_shift, axis=0)
    mode_model = TranslationRegistrationModel(translation_mode)
    distance = [m.change_relative_to(mode_model) for m in models]
    mode_model.confidence = models[np.argmin(distance)].confidence
    aprint(f"{len(models) - len(models_shift)} models with NaN confidence found.")
    aprint(f"Median model confidence of {mode_model.confidence}")
    return mode_model
