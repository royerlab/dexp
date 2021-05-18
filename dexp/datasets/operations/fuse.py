from typing import Sequence, List

from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed
from zarr.errors import ContainsArrayError, ContainsGroupError

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.multiview_lightsheet.fusion.mvsols import msols_fuse_1C2L
from dexp.processing.multiview_lightsheet.fusion.simview import simview_fuse_2C2L
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
                 huge_dataset,
                 workers,
                 workersbackend,
                 devices,
                 check,
                 stop_at_exception=True):

    views = tuple(dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True) for channel in channels)

    with asection(f"views:"):
        for view, channel in zip(views, channels):
            aprint(f"View: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    if slicing is not None:
        aprint(f"Slicing with: {slicing}")
        views = tuple(view[slicing] for view in views)

    # shape and dtype of views to fuse:
    shape = views[0].shape
    dtype = views[0].dtype
    nb_timepoints = shape[0]

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
            models = [None, ] * nb_timepoints

    # hold equalisation ratios:
    equalisation_ratios_reference: List[Sequence[float]] = [[]]
    if microscope == 'simview':
        equalisation_ratios_reference[0] = [None, None, None]
    elif microscope == 'mvsols':
        equalisation_ratios_reference[0] = [None, ]

    def process(tp, device, workers):
        try:

            with asection(f"Loading channels {channels} for time point {tp}/{nb_timepoints}"):
                views_tp = tuple(view[tp].compute() for view in views)

            with BestBackend(device, exclusive=True, enable_unified_memory=True):

                model = models[tp]

                # If we don't have a model for that timepoint we load one from a previous timepoint
                if model is None and tp >= workers:
                    aprint(f"we don't have a registration model for timepoint {tp}/{nb_timepoints} so we load one from a previous timepoint: {tp - workers}")
                    model = models[tp - workers]

                if microscope == 'simview':
                    tp_array, model, new_equalisation_ratios = simview_fuse_2C2L(*views_tp,
                                                                                 registration_force_model=loadreg,
                                                                                 registration_model=model,
                                                                                 registration_min_confidence=min_confidence,
                                                                                 registration_max_change=max_change,
                                                                                 registration_edge_filter=registration_edge_filter,
                                                                                 equalise=equalise,
                                                                                 equalisation_ratios=equalisation_ratios_reference[0],
                                                                                 zero_level=zero_level,
                                                                                 clip_too_high=clip_too_high,
                                                                                 fusion=fusion,
                                                                                 fusion_bias_exponent=2,
                                                                                 fusion_bias_strength_i=fusion_bias_strength_i,
                                                                                 fusion_bias_strength_d=fusion_bias_strength_d,
                                                                                 dehaze_size=dehaze_size,
                                                                                 dark_denoise_threshold=dark_denoise_threshold,
                                                                                 huge_dataset_mode=huge_dataset)
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
                                                                               equalisation_ratios=equalisation_ratios_reference[0],
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
                                                                               huge_dataset_mode=huge_dataset)

                with asection(f"Moving array from backend to numpy."):
                    tp_array = Backend.to_numpy(tp_array, dtype=dtype, force_copy=False)

                models[tp] = model.to_numpy()

                aprint(f"Last equalisation ratios: {new_equalisation_ratios}")
                if equalise_mode == 'first' and tp == 0:
                    aprint(f"Equalisation mode: 'first' -> saving equalisation ratios: {new_equalisation_ratios} for subsequent time points")
                    equalisation_ratios_reference[0] = list(float(Backend.to_numpy(ratio)) for ratio in new_equalisation_ratios)
                elif equalise_mode == 'all':
                    aprint(f"Equalisation mode: 'all' -> recomputing equalisation ratios for each time point.")
                    # No need to save, we need to recompute the ratios for each time point.
                    pass

            if 'fused' not in dest_dataset.channels():
                try:
                    dest_dataset.add_channel('fused',
                                             shape=(nb_timepoints,) + tp_array.shape,
                                             dtype=dtype,
                                             codec=compression,
                                             clevel=compression_level)
                except (ContainsArrayError, ContainsGroupError):
                    aprint(f"Other thread/process created channel before... ")

            with asection(f"Saving fused stack for time point {tp}, shape:{tp_array.shape}, dtype:{tp_array.dtype}"):
                dest_dataset.write_stack(channel='fused',
                                         time_point=tp,
                                         stack_array=tp_array)

            aprint(f"Done processing time point: {tp}/{nb_timepoints} .")

        except Exception as error:
            aprint(error)
            aprint(f"Error occurred while processing time point {tp} !")
            import traceback
            traceback.print_exc()

            if stop_at_exception:
                raise error

    if workers == -1:
        workers = len(devices)
    aprint(f"Number of workers: {workers}")

    if workers > 1:
        Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp, devices[tp % len(devices)], workers) for tp in range(0, nb_timepoints))
    else:
        for tp in range(0, nb_timepoints):
            process(tp, devices[0], workers)

    if not loadreg:
        model_list_to_file(model_list_filename, models)

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    dest_dataset.set_cli_history(parent=dataset if isinstance(dataset, ZDataset) else None)
    # close destination dataset:
    dest_dataset.close()
