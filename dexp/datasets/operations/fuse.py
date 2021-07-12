import dask
import numpy as np
from arbol.arbol import aprint
from arbol.arbol import asection
from dask.distributed import Client
import warnings

from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.multiview_lightsheet.fusion.mvsols import msols_fuse_1C2L
from dexp.processing.multiview_lightsheet.fusion.simview import SimViewFusion
from dexp.processing.registration.model.model_io import model_list_from_file, model_list_to_file
from dexp.utils.slicing import slice_from_shape


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

    dtype = views[0].dtype
    in_nb_time_points = views[0].shape[0]
    aprint(f"Slicing with: {slicing}")
    out_shape, volume_slicing, time_points = slice_from_shape(views[0].shape, slicing)

    if microscope == 'simview' and len(views) != 4:
        assert len(views) == 2
        warnings.warn('Only two views found. Fusing assuming they are from the same camera.')
        loadreg = False  # it does not have registration

    # load registration models:
    with NumpyBackend():
        if loadreg:
            aprint(f"Loading registration shifts from existing file! ({model_list_filename})")
            models = model_list_from_file(model_list_filename)
            if len(models) == 1:
                models = [models[0] for _ in range(in_nb_time_points)]
            elif len(models) != in_nb_time_points:
                raise ValueError(f'Number of registration models provided ({len(models)})'
                                 f'differs from number of input time points ({in_nb_time_points})')
        else:
            models = [None] * len(time_points)

    @dask.delayed
    def process(i, params):
        equalisation_ratios_reference, model, dest_dataset = params
        tp = time_points[i]
        try:
            with asection(f'Fusing time point for time point {i}/{len(time_points)}'):
                with asection(f"Loading channels {channels}"):
                    views_tp = tuple(np.asarray(view[tp][volume_slicing]) for view in views)

                with BestBackend(exclusive=True, enable_unified_memory=True):
                    if models[i] is not None:
                        model = models[i]
                    # otherwise it could be a None model or from the first iteration if equalisation mode was 'first'

                    if microscope == 'simview':
                        fuse_obj = SimViewFusion(registration_model=model,
                                                 equalise=equalise,
                                                 equalisation_ratios=equalisation_ratios_reference,
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
                                                                                   equalisation_ratios=equalisation_ratios_reference,
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
                        model = model.to_numpy()
                    else:
                        raise NotImplementedError

                    with asection(f"Moving array from backend to numpy."):
                        tp_array = Backend.to_numpy(tp_array, dtype=dtype, force_copy=False)

                    del views_tp
                    Backend.current().clear_memory_pool()

                    aprint(f"Last equalisation ratios: {new_equalisation_ratios}")

                with asection(f"Saving fused stack for time point {i}, shape:{tp_array.shape}, dtype:{tp_array.dtype}"):

                    if i == 0:
                        # We allocate last minute once we know the shape... because we don't always know the shape in advance!!!
                        mode = 'w' + ('' if overwrite else '-')
                        dest_dataset = ZDataset(output_path, mode, store, parent=dataset)
                        dest_dataset.add_channel('fused',
                                                 shape=(len(time_points),)+tp_array.shape,
                                                 dtype=tp_array.dtype,
                                                 codec=compression,
                                                 clevel=compression_level)

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

        return new_equalisation_ratios, model, dest_dataset

    from dask_cuda import LocalCUDACluster
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=devices)
    client = Client(cluster)
    aprint('Dask Client', client)

    # the parameters are (equalisation rations, registration model, dest. dataset)
    if microscope == 'simview':
        init_params = ([None, None, None], None, None)
    elif microscope == 'mvsols':
        init_params = ([None], None, None)
    else:
        raise NotImplementedError

    # it creates the output dataset from the first time point output shape
    params = process(0, init_params).persist()
    if equalise_mode == 'all':
        params = init_params[:2] + (params[2],)

    lazy_computations = []
    for i in range(1, len(time_points)):
        lazy_computations.append(process(i, params))

    models = [model for _, model, _ in dask.compute(*lazy_computations)]

    if not loadreg and models[0] is not None:
        model_list_to_file(model_list_filename, models)

    dest_dataset = params.compute()[2]

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
    client.close()
