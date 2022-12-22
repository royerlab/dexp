from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import dask
import numpy as np
from arbol.arbol import aprint, asection
from toolz import curry

from dexp.datasets import BaseDataset, ZDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.processing.multiview_lightsheet.fusion.mvsols import msols_fuse_1C2L
from dexp.processing.multiview_lightsheet.fusion.simview import SimViewFusion
from dexp.processing.registration.model.model_io import (
    model_list_from_file,
    model_list_to_file,
)
from dexp.processing.registration.model.pairwise_registration_model import (
    PairwiseRegistrationModel,
)
from dexp.utils import xpArray
from dexp.utils.backends import Backend, BestBackend
from dexp.utils.dask import get_dask_client


def load_registration_models(model_list_filename: Path, n_time_pts: int) -> Sequence[PairwiseRegistrationModel]:
    aprint(f"Loading registration shifts from existing file! ({model_list_filename})")
    models = model_list_from_file(model_list_filename)
    if len(models) == 1:
        models = [models[0] for _ in range(n_time_pts)]
    elif len(models) != n_time_pts:
        raise ValueError(
            f"Number of registration models provided ({len(models)})"
            f"differs from number of input time points ({n_time_pts})"
        )
    return models


@curry
def get_fusion_func(
    model: Optional[PairwiseRegistrationModel],
    microscope: str,
    metadata: Dict,
    equalisation_ratios: Optional[List],
    equalise: bool,
    zero_level: float,
    clip_too_high: float,
    fusion: str,
    fusion_bias_strength_i: float,
    fusion_bias_strength_d: float,
    dehaze_size: int,
    dark_denoise_threshold: int,
    z_pad_apodise: Tuple[int, int],
    loadreg: bool,
    warpreg_num_iterations: int,
    min_confidence: float,
    max_change: float,
    registration_edge_filter: bool,
    maxproj: bool,
    huge_dataset: bool,
    pad: bool,
    white_top_hat_size: float,
    white_top_hat_sampling: int,
    remove_beads: bool,
) -> Callable[[Dict], Tuple[xpArray, List, PairwiseRegistrationModel]]:

    if microscope == "simview":
        if equalisation_ratios is None:
            equalisation_ratios = [None, None, None]

        def fusion_func(views: Dict) -> Tuple:
            fuse_obj = SimViewFusion(
                registration_model=model,
                equalise=equalise,
                equalisation_ratios=equalisation_ratios,
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
                remove_beads=remove_beads,
                butterworth_filter_cutoff=1,
                flip_camera1=True,
                pad=pad,
            )

            stack = fuse_obj(**views)
            new_equalisation_ratios = fuse_obj._equalisation_ratios
            return stack, new_equalisation_ratios, model

    elif microscope == "mvsols":
        angle = metadata["angle"]
        dz = metadata["dz"]
        res = metadata["res"]
        illumination_correction_sigma = metadata["ic_sigma"] if "ic_sigma" in metadata else None
        if equalisation_ratios is None:
            equalisation_ratios = [None]

        def fusion_func(views: Dict) -> Tuple:
            stack, new_model, new_equalisation_ratios = msols_fuse_1C2L(
                *list(views.values()),
                z_pad=z_pad_apodise[0],
                z_apodise=z_pad_apodise[1],
                registration_num_iterations=warpreg_num_iterations,
                registration_force_model=loadreg,
                registration_model=model,
                registration_min_confidence=min_confidence,
                registration_max_change=max_change,
                registration_edge_filter=registration_edge_filter,
                equalise=equalise,
                equalisation_ratios=equalisation_ratios,
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
            return stack, new_equalisation_ratios, new_model

    else:
        raise NotImplementedError

    return fusion_func


@curry
def _process(
    time_point: int,
    views: Dict[str, StackIterator],
    out_dataset: ZDataset,
    fusion_func: Callable,
) -> Tuple[List, PairwiseRegistrationModel]:

    stack = list(views.values())[0]
    in_shape = stack.shape
    dtype = stack.dtype

    with asection(f"Fusing time point for time point {time_point}/{in_shape[0]}"):
        with asection(f"Loading channels {list(views.keys())}"):
            views_tp = {k: np.asarray(view[time_point]) for k, view in views.items()}

        with BestBackend():
            stack, new_equalisation_ratios, model = fusion_func(views_tp)

            with asection("Moving array from backend to numpy."):
                stack = Backend.to_numpy(stack, dtype=dtype, force_copy=False)

            aprint(f"Last equalisation ratios: {new_equalisation_ratios}")

            # moving to numpy to allow pickling
            if model is not None:
                model = model.to_numpy()

            new_equalisation_ratios = [None if v is None else Backend.to_numpy(v) for v in new_equalisation_ratios]

    with asection(f"Saving fused stack for time point {time_point}"):

        if time_point == 0:
            # We allocate last minute once we know the shape... because we don't always know
            # the shape in advance!!
            # @jordao NOTE: that could be pre computed
            out_dataset.add_channel("fused", shape=(in_shape[0],) + stack.shape, dtype=stack.dtype)

        out_dataset.write_stack(channel="fused", time_point=time_point, stack_array=stack)

    aprint(f"Done processing time point: {time_point} / {in_shape[0]}.")

    return new_equalisation_ratios, model


def dataset_fuse(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    microscope: str,
    model_list_filename: str,
    channels: Sequence[str],
    equalise: bool,
    equalise_mode: str,
    zero_level: float,
    clip_too_high: float,
    fusion: str,
    fusion_bias_strength_i: float,
    fusion_bias_strength_d: float,
    dehaze_size: int,
    dark_denoise_threshold: int,
    z_pad_apodise: Tuple[int, int],
    loadreg: bool,
    warpreg_num_iterations: int,
    min_confidence: float,
    max_change: float,
    registration_edge_filter: bool,
    maxproj: bool,
    huge_dataset: bool,
    pad: bool,
    white_top_hat_size: float,
    white_top_hat_sampling: int,
    remove_beads: bool,
    devices: Sequence[int],
):

    views = {channel.split("-")[-1]: input_dataset[channel] for channel in channels}
    n_time_pts = len(list(views.values())[0])

    with asection("Views:"):
        for channel, view in views.items():
            aprint(f"View: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    if microscope == "simview":
        views = SimViewFusion.validate_views(views)

    if loadreg:
        models = load_registration_models(model_list_filename, n_time_pts)

    # returns partial fusion function
    fusion_func = get_fusion_func(
        microscope=microscope,
        metadata=input_dataset.get_metadata(),
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
        loadreg=loadreg,
        white_top_hat_size=white_top_hat_size,
        white_top_hat_sampling=white_top_hat_sampling,
        remove_beads=remove_beads,
    )

    # it creates the output dataset from the first time point output shape
    equalisation_ratios, model = _process(
        time_point=0,
        views=views,
        out_dataset=output_dataset,
        fusion_func=fusion_func(
            model=models[0] if loadreg else None,
            equalisation_ratios=None,
        ),
    )

    if equalise_mode == "all":
        equalisation_ratios = None

    output_models = [model]

    client = get_dask_client(devices)
    aprint("Dask Client", client)

    lazy_computations = []
    for t in range(1, n_time_pts):
        lazy_computations.append(
            dask.delayed(_process)(
                time_point=t,
                views=views,
                out_dataset=output_dataset,
                fusion_func=fusion_func(equalisation_ratios=equalisation_ratios, model=models[t] if loadreg else None),
            )
        )

    # compute remaining stacks and save models
    output_models += [output[1] for output in dask.compute(*lazy_computations)]

    if not loadreg and output_models[0] is not None:
        model_list_to_file(model_list_filename, output_models)

    aprint(output_dataset.info())

    output_dataset.check_integrity()
