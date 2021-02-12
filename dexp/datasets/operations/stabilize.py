from typing import Sequence

from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed

from dexp.datasets.base_dataset import BaseDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.registration.sequence_proj import sequence_stabilisation_proj_


def dataset_stabilize(input_dataset: BaseDataset,
                      output_path: str,
                      channels: Sequence[str],
                      slicing=None,
                      zarr_store: str = 'dir',
                      compression_codec: str = 'zstd',
                      compression_level: int = 3,
                      overwrite: bool = False,
                      min_confidence: float = 0.3,
                      pad: bool = True,
                      workers: int = -1,
                      workers_backend: str = 'threading',
                      devices: Sequence[int] = (0,),
                      check: bool = True,
                      stop_at_exception: bool = True):
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
    min_confidence: minimal confidence below which pairwise registrations are rejected for the stabilisation.
    pad: pad input dataset.
    workers: number of workers, if -1 then the number of workers == numberof devices
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

        # Shape and chunks for array:
        shape = array.shape
        dtype = array.dtype
        chunks = ZDataset._default_chunks

        # Perform slicing:
        if slicing is not None:
            array = array[slicing]

        # Obtain projections:
        projections = []
        for axis in range(array.ndim - 1):
            projection = input_dataset.get_projection_array(channel=channel, axis=axis, wrap_with_dask=False)
            proj_axis = list(1 + a for a in range(array.ndim - 1) if a != axis)
            projections.append((*proj_axis, projection))

        # Perform stabilisation:
        model = sequence_stabilisation_proj_(projections=projections,
                                             min_confidence=min_confidence,
                                             ndim=3)

        # Shape of the resulting array:
        padded_shape = (shape[0],) + model.padded_shape(shape[1:])

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
                with asection(f"Loading channel: {channel} for time point {tp}"):
                    tp_array = array[tp].compute()

                with CupyBackend(device, exclusive=True):
                    with asection(f"Moving array to backend from numpy."):
                        tp_array = Backend.to_backend(tp_array)

                    with asection(f"Applying model..."):
                        tp_array = model.apply(tp_array, index=tp, pad=pad)

                    with asection(f"Moving array from backend to numpy."):
                        tp_array = Backend.to_numpy(tp_array, dtype=array.dtype, force_copy=False)

                with asection(f"Saving stabilized stack for time point {tp}, shape:{array.shape}, dtype:{array.dtype}"):
                    output_dataset.write_stack(channel=channel,
                                               time_point=tp,
                                               stack_array=tp_array)

                aprint(f"Done processing time point: {tp} .")

            except Exception as error:
                aprint(error)
                aprint(f"Error occurred while processing time point {tp} !")
                import traceback
                traceback.print_exc()
                if stop_at_exception:
                    raise error

        # Set number of workers:
        if workers == -1:
            workers = len(devices)
        aprint(f"Number of workers: {workers}")

        # start jobs:
        if workers > 1:
            Parallel(n_jobs=workers, backend=workers_backend)(delayed(process)(tp, devices[tp % len(devices)]) for tp in range(0, shape[0]))
        else:
            for tp in range(0, shape[0]):
                process(tp, devices[0])

    # printout output dataset info:
    aprint(output_dataset.info())
    if check:
        output_dataset.check_integrity()

    # close destination dataset:
    output_dataset.close()
