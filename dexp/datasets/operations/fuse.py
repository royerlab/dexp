import cupy
import numpy as np
from typing import List

from arbol.arbol import aprint
from arbol.arbol import asection
from zarr.errors import ContainsArrayError, ContainsGroupError

import dask
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.multiview_lightsheet.fusion.mvsols import msols_fuse_1C2L
from dexp.processing.multiview_lightsheet.fusion.simview import SimViewFusion
from dexp.processing.registration.model.model_io import model_list_from_file, model_list_to_file


def dataset_fuse(dataset,
                 output_path,
                 channels,
                 slicing,
                 store,
                 compression,
                 compression_level,
                 overwrite,
                 microscope,
                 equalise,
                 equalise_mode,
                 zero_level,
                 clip_too_high,
                 fusion,
                 fusion_bias_strength_i,
                 fusion_bias_strength_d,
                 dehaze_size,
                 dark_denoise_threshold,
                 z_pad_apodise,
                 loadreg,
                 model_list_filename,
                 warpreg_num_iterations,
                 min_confidence,
                 max_change,
                 registration_edge_filter,
                 maxproj,
                 huge_dataset,
                 workers,
                 workersbackend,
                 devices,
                 check,
                 stop_at_exception=True):

    views = tuple(dataset.get_array(channel, per_z_slice=False) for channel in channels)

    with asection(f"Views:"):
        for view, channel in zip(views, channels):
            aprint(f"View: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    output_shape = views[0][slicing].shape
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

    dtype = views[0].dtype

    # We allocate last minute once we know the shape...
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(output_path, mode, store)

    # load registration models:
    with NumpyBackend():
        if loadreg:
            aprint(f"Loading registration shifts from existing file! ({model_list_filename})")
            models = model_list_from_file(model_list_filename)
        else:
            models = [None] * len(time_points)

    # hold equalisation ratios:
    # equalisation_ratios_reference: List[List[float]] = [[]]
    # if microscope == 'simview':
    #     equalisation_ratios_reference[0] = [None, None, None]
    # elif microscope == 'mvsols':
    #     equalisation_ratios_reference[0] = [None, ]
    # else:
    #     raise NotImplementedError

    print('creating dataset', output_shape)
    if 'fused' not in dest_dataset.channels():
        try:
            dest_dataset.add_channel('fused',
                                     shape=output_shape,
                                     dtype=dtype,
                                     codec=compression,
                                     clevel=compression_level)
        except (ContainsArrayError, ContainsGroupError):
                aprint(f"Other thread/process created channel before... ")
    print('done')

    @dask.delayed
    def process(i):
        tp = time_points[i]
        try:
            with asection(f"Loading channels {channels} for time point {i}/{len(time_points)}"):
                views_tp = tuple(np.asarray(view[tp][slicing]) for view in views)

            with BestBackend(exclusive=True, enable_unified_memory=True):
                model = models[i]

                # If we don't have a model for that timepoint we load one from a previous timepoint

                # FIXME
                # if model is None and i >= workers:
                #     aprint(f"we don't have a registration model for timepoint {i}/{len(time_points)}"
                #            f"so we load one from a previous timepoint: {i - workers}")
                #     model = models[i - workers]

                if microscope == 'simview':
                    fuse_obj = SimViewFusion(registration_model=model,
                                             equalise=equalise,
                                             equalisation_ratios=[None, None, None],  # FIXME equalisation_ratios_reference[0],
                                             zero_level=zero_level,
                                             clip_too_high=clip_too_high,
                                             fusion=fusion,
                                             fusion_bias_exponent=2,
                                             fusion_bias_strength_i=fusion_bias_strength_i,
                                             fusion_bias_strength_d=fusion_bias_strength_d,
                                             dehaze_before_fusion=True,
                                             dehaze_size=dehaze_size,
                                             dehaze_correct_max_level=True,
                                             dark_denoise_threshold=dark_denoise_threshold,
                                             dark_denoise_size=9,
                                             butterworth_filter_cutoff=1,
                                             flip_camera1=True)

                    tp_array = fuse_obj(*views_tp)
                    new_equalisation_ratios = fuse_obj._equalisation_ratios

                elif microscope == 'mvsols':
                    metadata = dataset.get_metadata()
                    angle = metadata['angle']
                    channel = metadata['channel']
                    dz = metadata['dz']
                    res = metadata['res']
                    illumination_correction_sigma = metadata['ic_sigma'] if 'ic_sigma' in metadata else None

                    tp_array, model, new_equalisation_ratios = msols_fuse_1C2L(*views_tp,
                                                                               z_pad=z_pad_apodise[0],
                                                                               z_apodise=z_pad_apodise[1],
                                                                               registration_num_iterations=warpreg_num_iterations,
                                                                               registration_force_model=loadreg,
                                                                               registration_model=model,
                                                                               registration_min_confidence=min_confidence,
                                                                               registration_max_change=max_change,
                                                                               registration_edge_filter=registration_edge_filter,
                                                                               equalise=equalise,
                                                                               equalisation_ratios=None,  # FIXME equalisation_ratios_reference[0],
                                                                               zero_level=zero_level,
                                                                               clip_too_high=clip_too_high,
                                                                               fusion=fusion,
                                                                               fusion_bias_exponent=2,
                                                                               fusion_bias_strength_x=fusion_bias_strength_i,
                                                                               dehaze_size=dehaze_size,
                                                                               dark_denoise_threshold=dark_denoise_threshold,
                                                                               angle=angle,
                                                                               dx=res,
                                                                               dz=dz,
                                                                               illumination_correction_sigma=illumination_correction_sigma,
                                                                               registration_mode='projection' if maxproj else 'full',
                                                                               huge_dataset_mode=huge_dataset)
                else:
                    raise NotImplementedError

                with asection(f"Moving array from backend to numpy."):
                    tp_array = Backend.to_numpy(tp_array, dtype=dtype, force_copy=False)

                model = model.to_numpy()
                del views_tp
                Backend.current().clear_memory_pool()

                aprint(f"Last equalisation ratios: {new_equalisation_ratios}")
                # FIXME
                # if equalise_mode == 'first' and i == 0:
                #     aprint(f"Equalisation mode: 'first' -> saving equalisation ratios: {new_equalisation_ratios} for subsequent time points")
                #     if isinstance(new_equalisation_ratios[0], (np.ndarray, cupy.ndarray)) or None not in new_equalisation_ratios:
                #         equalisation_ratios_reference[0] = list(float(Backend.to_numpy(ratio)) for ratio in new_equalisation_ratios)
                # elif equalise_mode == 'all':
                #     aprint(f"Equalisation mode: 'all' -> recomputing equalisation ratios for each time point.")
                #     # No need to save, we need to recompute the ratios for each time point.
                #     pass

            with asection(f"Saving fused stack for time point {i}, shape:{tp_array.shape}, dtype:{tp_array.dtype}"):
                dest_dataset.write_stack(channel='fused',
                                         time_point=i,
                                         stack_array=tp_array)

            aprint(f"Done processing time point: {i}/{len(time_points)} .")

        except Exception as error:
            aprint(error)
            aprint(f"Error occurred while processing time point {i} !")
            import traceback
            traceback.print_exc()

            if stop_at_exception:
                raise error

        return model

    # if workers == -1:
    #     workers = len(devices)
    # aprint(f"Number of workers: {workers}")

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=devices)
    client = Client(cluster)
    aprint('Dask Client', client)

    print(time_points)
    lazy_computations = []
    for i in range(len(time_points)):
        lazy_computations.append(process(i))

    models = dask.compute(*lazy_computations)

    if not loadreg:
        model_list_to_file(model_list_filename, models)

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    dest_dataset.set_cli_history(parent=dataset if isinstance(dataset, ZDataset) else None)
    # close destination dataset:
    dest_dataset.close()
