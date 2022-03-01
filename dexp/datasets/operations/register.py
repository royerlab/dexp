from typing import List

import dask
import numpy
import numpy as np
from arbol.arbol import aprint, asection

from dexp.processing.multiview_lightsheet.fusion.simview import SimViewFusion
from dexp.processing.registration.model.model_io import model_list_to_file
from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.utils.backends import Backend, BestBackend
from dexp.utils.dask import get_dask_client


def dataset_register(
    dataset,
    model_path,
    channels,
    microscope,
    equalise,
    zero_level,
    clip_too_high,
    fusion,
    fusion_bias_strength_i,
    dehaze_size,
    registration_edge_filter,
    max_proj,
    white_top_hat_size,
    white_top_hat_sampling,
    devices,
    stop_at_exception=True,
):

    views = {channel.split("-")[-1]: dataset[channel] for channel in channels}

    with asection("Views:"):
        for channel, view in views.items():
            aprint(f"View: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    key = list(views.keys())[0]
    n_time_pts = len(views[key])

    if microscope == "simview":
        views = SimViewFusion.validate_views(views)

    @dask.delayed
    def process(i):
        try:
            with asection(f"Loading channels {channel} for time point {i}/{n_time_pts}"):
                views_tp = {k: np.asarray(view[i]) for k, view in views.items()}

            with BestBackend(exclusive=True, enable_unified_memory=True):
                if microscope == "simview":
                    fuse_obj = SimViewFusion(
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
                        flip_camera1=True,
                    )

                    C0Lx, C1Lx = fuse_obj.preprocess(**views_tp)
                    del views_tp
                    Backend.current().clear_memory_pool()
                    fuse_obj.compute_registration(
                        C0Lx,
                        C1Lx,
                        mode="projection" if max_proj else "full",
                        edge_filter=registration_edge_filter,
                        crop_factor_along_z=0.3,
                    )
                    del C0Lx, C1Lx
                    Backend.current().clear_memory_pool()
                    model = fuse_obj.registration_model.to_numpy()
                else:
                    raise NotImplementedError

            aprint(f"Done processing time point: {i}/{n_time_pts} .")

        except Exception as error:
            aprint(error)
            aprint(f"Error occurred while processing time point {i} !")
            import traceback

            traceback.print_exc()

            if stop_at_exception:
                raise error

        return model

    client = get_dask_client(devices)
    aprint("Dask Client", client)

    lazy_computations = []
    for i in range(n_time_pts):
        lazy_computations.append(process(i))

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
