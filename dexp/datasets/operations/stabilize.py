from typing import Optional, Sequence

import numpy
from arbol.arbol import aprint, asection
from joblib import Parallel, delayed
from toolz import curry

from dexp.datasets import BaseDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.datasets.zarr_dataset import ZDataset
from dexp.processing.registration.model.model_io import from_json
from dexp.processing.registration.model.sequence_registration_model import (
    SequenceRegistrationModel,
)
from dexp.processing.registration.sequence import image_stabilisation
from dexp.processing.registration.sequence_proj import image_stabilisation_proj_
from dexp.utils.backends import BestBackend, NumpyBackend
from dexp.utils.fft import clear_fft_plan_cache
from dexp.utils.misc import compute_num_workers


def _compute_model(
    input_dataset: BaseDataset,
    channel: str,
    max_range: int,
    min_confidence: float,
    enable_com: bool,
    quantile: float,
    tolerance: float,
    order_error: float,
    order_reg: float,
    alpha_reg: float,
    phase_correlogram_sigma: float,
    denoise_input_sigma: float,
    log_compression: bool,
    edge_filter: bool,
    detrend: bool = False,
    maxproj: bool = True,
    device: int = 0,
    workers: int = 1,
) -> SequenceRegistrationModel:

    # get channel array:
    array = input_dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

    # Perform slicing:
    if input_dataset.slicing is not None:
        # dexp slicing API doesn't work with dask
        array = array[input_dataset.slicing]

    # Shape and chunks for array:
    ndim = array.ndim

    workers = compute_num_workers(workers, array.shape[0])

    if not maxproj:
        with BestBackend(device, enable_unified_memory=True):
            model = image_stabilisation(
                image=array,
                axis=0,
                detrend=detrend,
                preload_images=False,
                max_range=max_range,
                min_confidence=min_confidence,
                enable_com=enable_com,
                quantile=quantile,
                tolerance=tolerance,
                order_error=order_error,
                order_reg=order_reg,
                alpha_reg=alpha_reg,
                sigma=phase_correlogram_sigma,
                denoise_input_sigma=denoise_input_sigma,
                log_compression=log_compression,
                edge_filter=edge_filter,
                internal_dtype=numpy.float16,
                workers=workers,
            )
            model.to_numpy()

    else:  # stabilized with maximum intensity projection
        projections = []
        for axis in range(array.ndim - 1):
            projection = input_dataset.get_projection_array(channel=channel, axis=axis, wrap_with_dask=False)
            if input_dataset.slicing is not None:
                projection = projection[input_dataset.slicing]

            proj_axis = list(1 + a for a in range(array.ndim - 1) if a != axis)
            projections.append((*proj_axis, projection))

        # Perform stabilisation:
        with BestBackend(device, enable_unified_memory=True):
            model = image_stabilisation_proj_(
                projections=projections,
                max_range=max_range,
                min_confidence=min_confidence,
                enable_com=enable_com,
                quantile=quantile,
                tolerance=tolerance,
                order_error=order_error,
                order_reg=order_reg,
                alpha_reg=alpha_reg,
                sigma=phase_correlogram_sigma,
                denoise_input_sigma=denoise_input_sigma,
                log_compression=log_compression,
                edge_filter=edge_filter,
                ndim=ndim - 1,
                internal_dtype=numpy.float16,
                detrend=detrend,
            )
            model.to_numpy()

    return model


# definition of function that processes each time point:
@curry
def _process(
    i: int,
    array: StackIterator,
    output_dataset: ZDataset,
    channel: str,
    model: SequenceRegistrationModel,
    pad: bool,
    integral: bool,
) -> None:
    with asection(f"Processing time point: {i}/{len(array)}"):
        with NumpyBackend() as bkd:
            with asection("Loading stack"):
                tp_array = bkd.to_backend(array[i])

                with asection("Applying model..."):
                    tp_array = model.apply(tp_array, index=i, pad=pad, integral=integral)

            with asection(
                f"Saving stabilized stack for time point {i}/{len(array)}, shape:{tp_array.shape}, "
                + "dtype:{array.dtype}"
            ):
                output_dataset.write_stack(channel=channel, time_point=i, stack_array=tp_array)
                clear_fft_plan_cache()


def dataset_stabilize(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    model_output_path: str,
    model_input_path: Optional[str] = None,
    reference_channel: Optional[str] = None,
    max_range: int = 7,
    min_confidence: float = 0.5,
    enable_com: bool = False,
    quantile: float = 0.5,
    tolerance: float = 1e-7,
    order_error: float = 2.0,
    order_reg: float = 1.0,
    alpha_reg: float = 0.1,
    phase_correlogram_sigma: float = 2,
    denoise_input_sigma: float = 1.5,
    log_compression: bool = True,
    edge_filter: bool = False,
    pad: bool = True,
    integral: bool = True,
    detrend: bool = False,
    maxproj: bool = True,
    workers: int = -1,
    workers_backend: str = "threading",
    device: int = 0,
):
    """
    Takes an input dataset and performs image stabilisation and outputs a stabilised dataset
    with given selected slice & channels in Zarr format with given store type, compression, etc...

    Parameters
    ----------
    input_dataset: Input dataset
    output_dataset: Output dataset
    channels: selected channels
    model_output_path: str
        Path to store computed stabilization model for reproducibility.
    model_input_path: Optional[str] = None,
        Path of previously computed stabilization model.
    reference_channel: use this channel to compute the stabilization model and apply to every channel.
    max_range: maximal distance, in time points, between pairs of images to registrate.
    min_confidence: minimal confidence below which pairwise registrations are rejected for the stabilisation.
    enable_com: enable center of mass fallback when standard registration fails.
    quantile: quantile to cut-off background in center-of-mass calculation
    tolerance: tolerance for linear solver.
    order_error: order for linear solver error term.
    order_reg: order for linear solver regularisation term.
    alpha_reg: multiplicative coefficient for regularisation term.
    phase_correlogram_sigma: sigma for Gaussian smoothing of phase correlogram, zero to disable.
    denoise_input_sigma : Uses a Gaussian filter to denoise input images, zero to disable.
    log_compression : Applies the function log1p to the images to compress high-intensities
        (usefull when very (too) bright structures are present in the images, such as beads).
    edge_filter : apply sobel edge filter to input images.
    pad: pad input dataset.
    integral: Set to True to only allow integral translations, False to allow subpixel accurate
        translations (induces blur!).
    detrend: removes linear detrend from stabilized image.
    maxproj: uses maximum intensity projection to compute the volume stabilization.
    workers: number of workers, if -1 then the number of workers == number of cores or devices
    workers_backend: What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread)
    device: Sets the CUDA devices id, e.g. 0,1,2
    """

    if model_input_path is not None and reference_channel is not None:
        raise ValueError("`model_input_path` and `reference channel` cannot be supplied at the same time.")

    model = None
    if model_input_path is not None:
        with open(model_input_path) as f:
            model = from_json(f.read())

    elif reference_channel is not None:
        if reference_channel not in input_dataset.channels():
            raise ValueError(f"Reference channel {reference_channel} not found.")
        model = _compute_model(
            input_dataset=input_dataset,
            channel=reference_channel,
            max_range=max_range,
            min_confidence=min_confidence,
            enable_com=enable_com,
            quantile=quantile,
            tolerance=tolerance,
            order_error=order_error,
            order_reg=order_reg,
            alpha_reg=alpha_reg,
            phase_correlogram_sigma=phase_correlogram_sigma,
            denoise_input_sigma=denoise_input_sigma,
            log_compression=log_compression,
            edge_filter=edge_filter,
            detrend=detrend,
            maxproj=maxproj,
            device=device,
            workers=workers,
        )
        with open(model_output_path, mode="w") as f:
            f.write(model.to_json())

    for channel in input_dataset._selected_channels(channels):
        if model is None:
            channel_model = _compute_model(
                input_dataset=input_dataset,
                channel=channel,
                max_range=max_range,
                min_confidence=min_confidence,
                enable_com=enable_com,
                quantile=quantile,
                tolerance=tolerance,
                order_error=order_error,
                order_reg=order_reg,
                alpha_reg=alpha_reg,
                phase_correlogram_sigma=phase_correlogram_sigma,
                denoise_input_sigma=denoise_input_sigma,
                log_compression=log_compression,
                edge_filter=edge_filter,
                detrend=detrend,
                maxproj=maxproj,
                device=device,
                workers=workers,
            )
            formatted_path = f"{model_output_path.split('.')[0]}_{channel}.json"
            with open(formatted_path, mode="w") as f:
                f.write(channel_model.to_json())

        else:
            channel_model = model

        metadata = input_dataset.get_metadata()
        if "dt" in metadata.get(channel, {}):
            prev_displacement = channel_model.total_displacement()
            channel_model = channel_model.reduce(int(round(metadata[channel]["dt"])))
            displacement = channel_model.total_displacement()

            assert len(displacement) == 3
            axes = ("tz", "ty", "tx")
            asection("Translation displacement after reduction")
            for name, prev, current in zip(axes, prev_displacement, displacement):
                # each displacement is a tuple (min, max) of displacement
                # minimum displacement different results in translations shift from reference channel
                metadata[channel][name] = current[0] - prev[0]
                aprint(f"{name.upper()}: {current[0] - prev[0]}")
            output_dataset.append_metadata(metadata)

        array = input_dataset[channel]
        shape = array.shape
        dtype = array.dtype
        nb_timepoints = shape[0]

        if len(channel_model) != nb_timepoints:
            aprint(
                f"WARNING: Number of time points ({nb_timepoints}) and registration models ({len(channel_model)})"
                f"does not match. Using the smallest one."
            )
            if nb_timepoints > len(channel_model):
                nb_timepoints = len(channel_model)
            # else: it is not necessary to change the model sequence length

        # Shape of the resulting array:
        padded_shape = (nb_timepoints,) + channel_model.padded_shape(shape[1:])
        output_dataset.add_channel(name=channel, shape=padded_shape, dtype=dtype)

        process = _process(
            array=array,
            output_dataset=output_dataset,
            channel=channel,
            model=channel_model,
            pad=pad,
            integral=integral,
        )

        # start jobs:
        if workers == 1:
            for tp in range(0, nb_timepoints):
                process(tp)
        else:
            n_jobs = compute_num_workers(workers, nb_timepoints)
            Parallel(n_jobs=n_jobs, backend=workers_backend)(delayed(process)(tp) for tp in range(0, nb_timepoints))

    # Dataset info:
    aprint(output_dataset.info())

    # Check dataset integrity:
    output_dataset.check_integrity()
