from typing import Dict, Optional, Sequence, Tuple

import dask
import numpy as np
from arbol.arbol import aprint, asection
from toolz import curry

from build.lib.dexp.datasets.base_dataset import BaseDataset
from dexp.datasets import ZDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.processing.multiview_lightsheet.fusion.mvsols import msols_fuse_1C2L
from dexp.processing.multiview_lightsheet.fusion.simview import SimViewFusion
from dexp.processing.registration.model import PairwiseRegistrationModel
from dexp.processing.registration.model.model_io import (
    model_list_from_file,
    model_list_to_file,
)
from dexp.utils.backends import Backend, BestBackend, NumpyBackend
from dexp.utils.dask import get_dask_client


@curry
def _process(
    index: int,
    fusion_params: Tuple,
    dataset: BaseDataset,
    views: Dict[str, StackIterator],
    models: Sequence[Optional[PairwiseRegistrationModel]],
    output_path: str,
    *,
    store,
    compression,
    compression_level,
    overwrite,
    microscope,
    equalise,
    zero_level,
    clip_too_high,
    fusion,
    fusion_bias_strength_i,
    fusion_bias_strength_d,
    dehaze_size,
    dark_denoise_threshold,
    z_pad_apodise,
    loadreg,
    warpreg_num_iterations,
    min_confidence,
    max_change,
    registration_edge_filter,
    maxproj,
    huge_dataset,
    pad,
    white_top_hat_size,
    white_top_hat_sampling,
    dtype,
    device_id: int = 0,
) -> Tuple:
    # FIXME: look how the parameters are provided multiple times, this should be curried or a class

    equalisation_ratios_reference, model, dest_dataset = fusion_params
    n_time_pts = len(next(views.values()))

    with asection(f"Fusing time point for time point {index}/{n_time_pts}"):
        with asection(f"Loading channels {views.keys()}"):
            views_tp = {k: np.asarray(view[index]) for k, view in views.items()}

        with BestBackend(exclusive=True, enable_unified_memory=True, device_id=device_id):
            if models[index] is not None:
                model = models[index]
            # otherwise it could be a None model or from the first iteration if equalisation mode was 'first'
            if microscope == "simview":
                fuse_obj = SimViewFusion(
                    registration_model=model,
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
                    white_top_hat_size=white_top_hat_size,
                    white_top_hat_sampling=white_top_hat_sampling,
                    butterworth_filter_cutoff=1,
                    flip_camera1=True,
                    pad=pad,
                )

                tp_array = fuse_obj(**views_tp)
                new_equalisation_ratios = fuse_obj._equalisation_ratios

            elif microscope == "mvsols":
                metadata = dataset.get_metadata()
                angle = metadata["angle"]
                dz = metadata["dz"]
                res = metadata["res"]
                illumination_correction_sigma = metadata["ic_sigma"] if "ic_sigma" in metadata else None

                tp_array, model, new_equalisation_ratios = msols_fuse_1C2L(
                    *list(views_tp.values()),
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
                    registration_mode="projection" if maxproj else "full",
                    huge_dataset_mode=huge_dataset,
                )
            else:
                raise NotImplementedError

            with asection("Moving array from backend to numpy."):
                tp_array = Backend.to_numpy(tp_array, dtype=dtype, force_copy=False)

            del views_tp
            Backend.current().clear_memory_pool()

            aprint(f"Last equalisation ratios: {new_equalisation_ratios}")

            # moving to numpy to allow pickling
            if model is not None:
                model = model.to_numpy()
            new_equalisation_ratios = [None if v is None else Backend.to_numpy(v) for v in new_equalisation_ratios]

        with asection(f"Saving fused stack for time point {index}, shape:{tp_array.shape}, dtype:{tp_array.dtype}"):

            if index == 0:
                # We allocate last minute once we know the shape... because we don't always know
                # the shape in advance!!!
                mode = "w" + ("" if overwrite else "-")
                dest_dataset = ZDataset(output_path, mode, store, parent=dataset)
                dest_dataset.add_channel(
                    "fused",
                    shape=(n_time_pts,) + tp_array.shape,
                    dtype=tp_array.dtype,
                    codec=compression,
                    clevel=compression_level,
                )

            dest_dataset.write_stack(channel="fused", time_point=index, stack_array=tp_array)

        aprint(f"Done processing time point: {index}/{n_time_pts} .")

    return new_equalisation_ratios, model, dest_dataset


def dataset_fuse(
    dataset,
    output_path,
    channels,
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
    devices,
    check,
    pad,
    white_top_hat_size,
    white_top_hat_sampling,
):

    views = {channel.split("-")[-1]: dataset[channel] for channel in channels}

    with asection("Views:"):
        for channel, view in views.items():
            aprint(f"View: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    key = list(views.keys())[0]
    dtype = views[key].dtype
    n_time_pts = len(views[key])

    if microscope == "simview":
        views = SimViewFusion.validate_views(views)

    # load registration models:
    with NumpyBackend():
        if loadreg:
            aprint(f"Loading registration shifts from existing file! ({model_list_filename})")
            models = model_list_from_file(model_list_filename)
            if len(models) == 1:
                models = [models[0] for _ in range(n_time_pts)]
            elif len(models) != n_time_pts:
                raise ValueError(
                    f"Number of registration models provided ({len(models)})"
                    f"differs from number of input time points ({n_time_pts})"
                )
        else:
            models = [None] * len(n_time_pts)

    process = _process(
        dataset=dataset,
        views=views,
        models=models,
        output_path=output_path,
        store=store,
        compression=compression,
        compression_level=compression_level,
        overwrite=overwrite,
        microscope=microscope,
        equalise=equalise,
        zero_level=zero_level,
        clip_too_high=clip_too_high,
        fusion=fusion,
        fusion_bias_strength_i=fusion_bias_strength_i,
        fusion_bias_strength_d=fusion_bias_strength_d,
        dehaze_size=dehaze_size,
        dark_denoise_threshold=dark_denoise_threshold,
        z_pad_apodise=z_pad_apodise,
        warpreg_num_iterations=warpreg_num_iterations,
        min_confidence=min_confidence,
        max_change=max_change,
        registration_edge_filter=registration_edge_filter,
        maxproj=maxproj,
        huge_dataset=huge_dataset,
        pad=pad,
        white_top_hat_size=white_top_hat_size,
        white_top_hat_sampling=white_top_hat_sampling,
        dtype=dtype,
    )

    if not isinstance(devices, str) and len(devices) == 1:
        process = process(device_id=devices[0])
    else:
        process = dask.delayed(process)
        client = get_dask_client(devices)
        aprint("Dask Client", client)

    # the parameters are (equalisation rations, registration model, dest. dataset)
    if microscope == "simview":
        init_params = ([None, None, None], None, None)
    elif microscope == "mvsols":
        init_params = ([None], None, None)
    else:
        raise NotImplementedError

    # it creates the output dataset from the first time point output shape
    params = process(0, init_params)
    if len(devices) > 1:
        params = params.persist()

    if equalise_mode == "all":
        params = init_params[:2] + (params[2],)

    lazy_computations = []  # it is only lazy if len(devices) > 1
    for i in range(1, len(n_time_pts)):
        lazy_computations.append(process(i, params))

    if len(devices) > 1:
        first_model, dest_dataset = params.compute()[1:]
        models = [first_model] + [output[1] for output in dask.compute(*lazy_computations)]
    else:
        models = [params[1]] + [output[1] for output in lazy_computations]
        dest_dataset = params[2]

    if not loadreg and models[0] is not None:
        model_list_to_file(model_list_filename, models)

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
    if len(devices) > 1:
        client.close()
