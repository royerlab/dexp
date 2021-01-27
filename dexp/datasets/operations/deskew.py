from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed
from zarr.errors import ContainsArrayError, ContainsGroupError

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.registration.model.model_io import model_list_to_file


def dataset_deskew(dataset,
                   output_path,
                   channels,
                   slicing,
                   store,
                   compression,
                   compression_level,
                   overwrite,
                   zero_level,
                   clip_too_high,
                   dehaze_size,
                   dark_denoise_threshold,
                   z_pad_apodise,
                   workers,
                   workersbackend,
                   devices,
                   check,
                   stop_at_exception=True):
    arrays = tuple(dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True) for channel in channels)

    with asection(f"views:"):
        for view, channel in zip(arrays, channels):
            aprint(f"Channel: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    if slicing is not None:
        aprint(f"Slicing with: {slicing}")
        arrays = tuple(view[slicing] for view in arrays)

    # We allocate last minute once we know the shape...
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(output_path, mode, store)

    def process(tp, device, workers):
        try:
            for array, channel in zip(arrays, channels):

                # shape and dtype of views to deskew:
                shape = array[0].shape
                nb_timepoints = array.shape[0]
                dtype = array[0].dtype

                with asection(f"Loading channel {channel} for time point {tp}"):
                    view_tp = array[tp].compute()

                with CupyBackend(device, exclusive=True, enable_unified_memory=True):

                    metadata = dataset.get_metadata()
                    angle = metadata['angle']
                    channel = metadata['channel']
                    dz = metadata['dz']
                    res = metadata['res']

                    deskewed_view_tp, = deskew(views_tp,
                                               zero_level=zero_level,
                                               clip_too_high=clip_too_high,
                                               dehaze_size=dehaze_size,
                                               dark_denoise_threshold=dark_denoise_threshold,
                                               angle=angle,
                                               dx=res,
                                               dz=dz)

                    with asection(f"Moving array from backend to numpy."):
                        deskewed_view_tp = Backend.to_numpy(deskewed_view_tp, dtype=dtype, force_copy=False)

                if 'fused' not in dest_dataset.channels():
                    try:
                        dest_dataset.add_channel('fused',
                                                 shape=(nb_timepoints,) + deskewed_view_tp.shape,
                                                 dtype=dtype,
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
            aprint(f"Error occurred while processing time point {tp} !")
            import traceback
            traceback.print_exc()

            if stop_at_exception:
                raise error

    if workers == -1:
        workers = len(devices)
    aprint(f"Number of workers: {workers}")

    if workers > 1:
        Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp, devices[tp % len(devices)], workers) for tp in range(0, shape[0]))
    else:
        for tp in range(0, shape[0]):
            process(tp, devices[0], workers)

    if not loadreg:
        model_list_to_file(model_list_filename, models)

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
