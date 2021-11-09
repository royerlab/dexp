from typing import List

import dask
import numpy
import numpy as np
from arbol.arbol import aprint, asection
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.multiview_lightsheet.fusion.simview import SimViewFusion
from dexp.processing.registration.model.model_io import model_list_to_file
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.utils.slicing import slice_from_shape


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

    views = {
        channel.split('-')[-1]: dataset.get_array(channel, per_z_slice=False)
        for channel in channels
    }

    with asection(f"Views:"):
        for channel, view in views.items():
            aprint(f"View: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    key = list(views.keys())[0]
    print(f"Slicing with: {slicing}")
    _, volume_slicing, time_points = slice_from_shape(views[key].shape, slicing)

    if microscope == 'simview':
        views = SimViewFusion.validate_views(views)
 
    @dask.delayed
    def process(i):
        tp = time_points[i]
        try:
            with asection(f"Loading channels {channel} for time point {i}/{len(time_points)}"):
                views_tp = {k: np.asarray(view[tp][volume_slicing]) for k, view in views.items()}

            with BestBackend(exclusive=True, enable_unified_memory=True):
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

                    C0Lx, C1Lx = fuse_obj.preprocess(**views_tp)
                    del views_tp
                    Backend.current().clear_memory_pool()
                    fuse_obj.compute_registration(C0Lx, C1Lx, mode='projection' if max_proj else 'full',
                                                  edge_filter=registration_edge_filter,
                                                  crop_factor_along_z=0.3)
                    del C0Lx, C1Lx
                    Backend.current().clear_memory_pool()
                    model = fuse_obj.registration_model.to_numpy()
                else:
                    raise NotImplementedError

            aprint(f"Done processing time point: {i}/{len(time_points)} .")

        except Exception as error:
            aprint(error)
            aprint(f"Error occurred while processing time point {i} !")
            import traceback
            traceback.print_exc()

            if stop_at_exception:
                raise error

        return model

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=devices)
    client = Client(cluster)
    aprint('Dask Client', client)

    lazy_computations = []
    for i in range(len(time_points)):
        lazy_computations.append(process(i))

    models = dask.compute(*lazy_computations)

    mode_model = compute_median_translation(models)
    model_list_to_file(model_path, [mode_model])
    client.close()


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
