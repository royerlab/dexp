import numpy
import numpy as np
from arbol.arbol import aprint, asection
from typing import List

from joblib import Parallel, delayed

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.registration.model.model_io import model_list_to_file
from dexp.processing.multiview_lightsheet.fusion.simview import SimViewFusion
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel


def dataset_register(dataset,
                     model_path,
                     channels,
                     slicing,
                     microscope,
                     equalise,
                     zero_level,
                     clip_too_high,
                     fusion,
                     fusion_bias_strength_i,
                     dehaze_size,
                     registration_edge_filter,
                     max_proj,
                     workers,
                     workers_backend,
                     devices,
                     stop_at_exception=True):

    views = tuple(dataset.get_array(channel, per_z_slice=False) for channel in channels)

    with asection(f"Views:"):
        for view, channel in zip(views, channels):
            aprint(f"View: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    total_time_points = views[0].shape[0]
    time_points = list(range(total_time_points))
    if slicing is not None:
        aprint(f"Slicing with: {slicing}")
        if isinstance(slicing, tuple):
            time_points = time_points[slicing[0]]
            slicing = slicing[1:]
        else:  # slicing only over time
            time_points = time_points[slicing]
            slicing = ...
    else:
        slicing = ...

    models = [None] * len(time_points)

    def process(i, device, workers):
        tp = time_points[i]
        try:
            with asection(f"Loading channels {channel} for time point {i}/{len(time_points)}"):
                views_tp = tuple(np.asarray(view[tp][slicing]) for view in views)

            with BestBackend(device, exclusive=True, enable_unified_memory=True):
                if microscope == 'simview':
                    fuse_obj = SimViewFusion(registration_model=None,
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
                                             flip_camera1=True)

                    C0Lx, C1Lx = fuse_obj.preprocess(*views_tp)
                    del views_tp
                    Backend.current().clear_memory_pool()
                    fuse_obj.compute_registration(C0Lx, C1Lx, mode='projection' if max_proj else 'full',
                                                  edge_filter=registration_edge_filter,
                                                  crop_factor_along_z=0.3)
                    del C0Lx, C1Lx
                    Backend.current().clear_memory_pool()
                else:
                    raise NotImplementedError

                models[i] = fuse_obj.registration_model.to_numpy()

            aprint(f"Done processing time point: {i}/{len(time_points)} .")

        except Exception as error:
            aprint(error)
            aprint(f"Error occurred while processing time point {i} !")
            import traceback
            traceback.print_exc()

            if stop_at_exception:
                raise error

    if workers == -1:
        workers = len(devices)
    aprint(f"Number of workers: {workers}")

    if workers > 1:
        parallel = Parallel(n_jobs=workers, backend=workers_backend)
        parallel(delayed(process)(i, devices[i % len(devices)], workers)
                 for i in range(len(time_points)))
    else:
        for i in range(len(time_points)):
            process(i, devices[0], workers)

    mode_model = compute_median_translation(models)
    model_list_to_file(model_path, [mode_model] * total_time_points)


def compute_median_translation(models: List[TranslationRegistrationModel]) \
        -> TranslationRegistrationModel:
    """
    Computes the median of each axis over all registration models and
    assigns the confidence to the one with smallest distance to it.
    """
    models_shift = np.vstack([m.shift_vector for m in models if not numpy.isnan(m.confidence)])
    translation_mode = np.median(models_shift, axis=0)
    mode_model = TranslationRegistrationModel(translation_mode)
    distance = [m.change_relative_to(mode_model) for m in models]
    mode_model.confidence = models[np.argmin(distance)].confidence
    aprint(f'{len(models) - len(models_shift)} models with NaN confidence found.')
    aprint(f"Median model confidence of {mode_model.confidence}")
    return mode_model
