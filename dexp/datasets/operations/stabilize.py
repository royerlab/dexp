import os
from typing import Sequence, Optional

import numpy
from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed

from dexp.datasets.base_dataset import BaseDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.sequence_proj import image_stabilisation_proj_
from dexp.processing.registration.sequence import image_stabilisation
from dexp.processing.registration.model.sequence_registration_model import SequenceRegistrationModel


def _compute_model(
        input_dataset: BaseDataset,
        channel: str,
        slicing,
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
        debug_output=None) -> SequenceRegistrationModel:

    # get channel array:
    array = input_dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

    # Perform slicing:
    if slicing is not None:
        array = array[slicing]

    # Shape and chunks for array:
    ndim = array.ndim

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
                debug_output=debug_output,
            )
            model.to_numpy()

    else:  # stabilized with maximum intensity projection
        projections = []
        for axis in range(array.ndim - 1):
            projection = input_dataset.get_projection_array(
                channel=channel, axis=axis, wrap_with_dask=False
            )
            if slicing is not None:
                projection = projection[slicing]

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
                debug_output=debug_output,
                detrend=detrend,
            )
            model.to_numpy()

    return model


def dataset_stabilize(input_dataset: BaseDataset,
                      output_path: str,
                      channels: Sequence[str],
                      reference_channel: Optional[str] = None,
                      slicing=None,
                      zarr_store: str = "dir",
                      compression_codec: str = "zstd",
                      compression_level: int = 3,
                      overwrite: bool = False,
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
                      check: bool = True,
                      stop_at_exception: bool = True,
                      debug_output=None):
    """
    Takes an input dataset and performs image stabilisation and outputs a stabilised dataset with given selected slice & channels in Zarr format with given store type, compression, etc...

    Parameters
    ----------
    input_dataset: Input dataset
    output_path: Output path for Zarr storage
    channels: selected channels
    reference_channel: use this channel to compute the stabilization model and apply to every channel.
    slicing: selected array slicing
    zarr_store: type of store, can be 'dir', 'ndir', or 'zip'
    compression_codec: compression codec to be used ('zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' or 'snappy').
    compression_level: An integer between 0 and 9 specifying the compression level.
    overwrite: overwrite output dataset if already exists
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
    log_compression : Applies the function log1p to the images to compress high-intensities (usefull when very (too) bright structures are present in the images, such as beads)
    edge_filter : apply sobel edge filter to input images.
    pad: pad input dataset.
    integral: Set to True to only allow integral translations, False to allow subpixel accurate translations (induces blur!).
    detrend: removes linear detrend from stabilized image.
    maxproj: uses maximum intensity projection to compute the volume stabilization.
    workers: number of workers, if -1 then the number of workers == number of cores or devices
    workers_backend: What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread)
    device: Sets the CUDA devices id, e.g. 0,1,2
    check: Checking integrity of written file.
    stop_at_exception: True to stop as soon as there is an exception during processing.
    """

    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    output_dataset = ZDataset(output_path, mode, zarr_store)

    model = None
    if reference_channel is not None:
        if reference_channel not in input_dataset.channels():
            raise ValueError(f'Reference channel {reference_channel} not found.')
        model = _compute_model(
            input_dataset=input_dataset,
            channel=reference_channel,
            slicing=slicing,
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
            debug_output=debug_output,
            )

    for channel in input_dataset._selected_channels(channels):
        if reference_channel is None:
            channel_model = _compute_model(
                input_dataset=input_dataset,
                channel=channel,
                slicing=slicing,
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
                debug_output=debug_output,
            )
        else:
            channel_model = model

        metadata = input_dataset.get_metadata()
        if 'dt' in metadata.get(channel, {}):
            channel_model = channel_model.reduce(int(round(metadata[channel]['dt'])))

        array = input_dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)
        if slicing is not None:
            array = array[slicing]

        shape = array.shape
        dtype = array.dtype
        nb_timepoints = shape[0]

        if len(channel_model) != nb_timepoints:
            aprint(f'WARNING: Number of time points ({nb_timepoints}) and registration models ({len(channel_model)})'
                   f'does not match. Using the smallest one.')
            if nb_timepoints > len(channel_model):
                nb_timepoints = len(channel_model)
            # else: it is not necessary to change the model sequence length

        # Shape of the resulting array:
        padded_shape = (nb_timepoints,) + channel_model.padded_shape(shape[1:])

        # Add channel to output dataset:
        output_dataset.add_channel(name=channel,
                                   shape=padded_shape,
                                   dtype=dtype,
                                   codec=compression_codec,
                                   clevel=compression_level)

        # definition of function that processes each time point:
        def process(tp):
            try:
                with asection(f"Processing time point: {tp}/{nb_timepoints} ."):
                    with asection(f"Loading stack"):
                        tp_array = array[tp].compute()

                    with NumpyBackend():
                        with asection(f"Applying model..."):
                            tp_array = channel_model.apply(tp_array, index=tp, pad=pad, integral=integral)

                    with asection(f"Saving stabilized stack for time point {tp}/{nb_timepoints}, shape:{tp_array.shape}, dtype:{array.dtype}"):
                        output_dataset.write_stack(channel=channel,
                                                   time_point=tp,
                                                   stack_array=tp_array)

            except Exception as error:
                aprint(error)
                aprint(f"Error occurred while processing time point {tp}/{nb_timepoints} !")
                import traceback
                traceback.print_exc()
                if stop_at_exception:
                    raise error

        # Set number of workers:
        if workers == -1:
            workers = max(1, os.cpu_count() // abs(workers))
        aprint(f"Number of workers: {workers}")

        # start jobs:
        if workers > 1:
            Parallel(n_jobs=workers, backend=workers_backend)(delayed(process)(tp) for tp in range(0, nb_timepoints))
        else:
            for tp in range(0, nb_timepoints):
                process(tp)

    # printout output dataset info:
    aprint(output_dataset.info())
    if check:
        output_dataset.check_integrity()

    output_dataset.set_cli_history(parent=input_dataset if isinstance(input_dataset, ZDataset) else None)
    # close destination dataset:
    output_dataset.close()
