import os
from typing import Sequence

import numpy
from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed

from dexp.datasets.base_dataset import BaseDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.sequence_proj import image_stabilisation_proj_


def dataset_stabilize(input_dataset: BaseDataset,
                      output_path: str,
                      channels: Sequence[str],
                      slicing=None,
                      zarr_store: str = 'dir',
                      compression_codec: str = 'zstd',
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
                      workers: int = -1,
                      workers_backend: str = 'threading',
                      devices: Sequence[int] = (0,),
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
    workers: number of workers, if -1 then the number of workers == number of cores or devices
    workers_backend: What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread)
    devices: Sets the CUDA devices id, e.g. 0,1,2 or ‘all’
    check: Checking integrity of written file.
    stop_at_exception: True to stop as soon as there is an exception during processing.
    """

    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    output_dataset = ZDataset(output_path, mode, zarr_store)

    for channel in input_dataset._selected_channels(channels):

        # get channel array:
        array = input_dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

        # Perform slicing:
        if slicing is not None:
            array = array[slicing]

        # Shape and chunks for array:
        ndim = array.ndim
        shape = array.shape
        dtype = array.dtype
        chunks = ZDataset._default_chunks
        nb_timepoints = shape[0]

        # Obtain projections:
        projections = []
        for axis in range(array.ndim - 1):
            projection = input_dataset.get_projection_array(channel=channel, axis=axis, wrap_with_dask=False)
            if slicing is not None:
                projection = projection[slicing]

            proj_axis = list(1 + a for a in range(array.ndim - 1) if a != axis)
            projections.append((*proj_axis, projection))

        # Perform stabilisation:
        with BestBackend(devices[0], enable_unified_memory=True):
            model = image_stabilisation_proj_(projections=projections,
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
                                              debug_output=debug_output)
            model.to_numpy()

        # Shape of the resulting array:
        padded_shape = (nb_timepoints,) + model.padded_shape(shape[1:])

        # Add channel to output datatset:
        output_dataset.add_channel(name=channel,
                                   shape=padded_shape,
                                   dtype=dtype,
                                   chunks=chunks,
                                   codec=compression_codec,
                                   clevel=compression_level)

        # definition of function that processes each time point:
        def process(tp, device):

            try:
                with asection(f"Processing time point: {tp}/{nb_timepoints} ."):
                    with asection(f"Loading stack"):
                        tp_array = array[tp].compute()

                    # backend = NumpyBackend() if integral else CupyBackend(device, exclusive=True)

                    with NumpyBackend():
                        xp = Backend.get_xp_module()

                        with asection(f"Applying model..."):
                            # tp_array = Backend.to_backend(tp_array, dtype=array.dtype, force_copy=False)
                            tp_array = model.apply(tp_array, index=tp, pad=pad, integral=integral)
                            # tp_array = Backend.to_numpy(tp_array, dtype=array.dtype, force_copy=False)

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
            # Note: for some reason, having two many threads running concurently seems to lead to a freeze. Unclear why.
            workers = max(4, os.cpu_count() // 4)
        aprint(f"Number of workers: {workers}")

        # start jobs:
        if workers > 1:
            Parallel(n_jobs=workers, backend=workers_backend)(delayed(process)(tp, devices[tp % len(devices)]) for tp in range(0, nb_timepoints))
        else:
            for tp in range(0, nb_timepoints):
                process(tp, devices[0])

    # printout output dataset info:
    aprint(output_dataset.info())
    if check:
        output_dataset.check_integrity()

    # close destination dataset:
    output_dataset.close()
