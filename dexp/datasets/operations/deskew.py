from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed
from zarr.errors import ContainsArrayError, ContainsGroupError

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.deskew.yang_deskew import yang_deskew


def dataset_deskew(dataset,
                   output_path,
                   channels,
                   slicing,
                   store,
                   compression,
                   compression_level,
                   flips,
                   overwrite,
                   workers,
                   workersbackend,
                   devices,
                   check,
                   stop_at_exception=True):

    arrays = tuple(dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True) for channel in channels)

    with asection(f"Channels:"):
        for view, channel in zip(arrays, channels):
            aprint(f"Channel: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    if slicing is not None:
        aprint(f"Slicing with: {slicing}")
        arrays = tuple(view[slicing] for view in arrays)

    # We allocate last minute once we know the shape...
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(output_path, mode, store)

    metadata = dataset.get_metadata()
    angle = metadata['angle']
    channel = metadata['channel']
    dz = metadata['dz']
    res = metadata['res']

    for array, channel, flip in zip(arrays, channels, flips):

        # shape and dtype of views to deskew:
        shape = array[0].shape
        nb_timepoints = array.shape[0]

        def process(channel, tp, device):
            try:

                with asection(f"Loading channel {channel} for time point {tp}"):
                    array_tp = array[tp].compute()

                with BestBackend(device, exclusive=True, enable_unified_memory=True):

                    if 'yang' in mode:
                        deskewed_view_tp, = yang_deskew(image=array_tp,
                                                       flip=flip,
                                                       angle=angle,
                                                       dx=res,
                                                       dz=dz)
                    elif 'classic' in mode:
                        raise NotImplementedError('Classic deskewing not yet implemented')
                    else:
                        raise ValueError(f"Deskew mode: {mode} not supported.")

                    with asection(f"Moving array from backend to numpy."):
                        deskewed_view_tp = Backend.to_numpy(deskewed_view_tp, dtype=array.dtype, force_copy=False)

                if channel not in dest_dataset.channels():
                    try:
                        dest_dataset.add_channel(channel,
                                                 shape=array.shape,
                                                 dtype=array.dtype,
                                                 codec=compression,
                                                 clevel=compression_level)
                    except (ContainsArrayError, ContainsGroupError):
                        aprint(f"Other thread/process created channel before... ")

                with asection(f"Saving fused stack for time point {tp}, shape:{deskewed_view_tp.shape}, dtype:{deskewed_view_tp.dtype}"):
                    dest_dataset.write_stack(channel='fused',
                                             time_point=tp,
                                             stack_array=deskewed_view_tp)

                aprint(f"Done processing time point: {tp} .")

            except Exception as error:
                aprint(error)
                aprint(f"Error occurred while deskewing time point {tp} !")
                import traceback
                traceback.print_exc()

                if stop_at_exception:
                    raise error

        if workers == -1:
            workers = len(devices)
        aprint(f"Number of workers: {workers}")

        if workers > 1:
            Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(channel, tp, devices[tp % len(devices)]) for tp in range(0, shape[0]))
        else:
            for tp in range(0, nb_timepoints):
                process(channel, tp, devices[0])


    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    dest_dataset.set_cli_history(parent=dataset if isinstance(dataset, ZDataset) else None)
    # close destination dataset:
    dest_dataset.close()
