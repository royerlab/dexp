from typing import Sequence

from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed

from dexp.datasets.base_dataset import BaseDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.registration.sequence_proj import sequence_stabilisation_proj_


def dataset_stabilize(dataset: BaseDataset,
                      path: str,
                      channels: Sequence[str],
                      slicing,
                      store: str,
                      compression: str,
                      compression_level: int,
                      overwrite: bool,
                      minconfidence: float,
                      workers: int,
                      workersbackend: str,
                      devices: Sequence[int],
                      check: bool,
                      stop_at_exception: bool = True):
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(path, mode, store)

    for channel in dataset._selected_channels(channels):

        array = dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

        shape = array.shape
        chunks = dataset._default_chunks

        if slicing is not None:
            array = array[slicing]

        projections = []
        for axis in range(array.ndim):
            xp = Backend.get_xp_module()
            projection = dataset.get_projection_array(channel=channel, axis=axis, wrap_with_dask=False)
            proj_axis = list(range(3)).remove(axis)
            projections.append((*proj_axis, projection))

        model = sequence_stabilisation_proj_(projections=projections,
                                             minconfidence=minconfidence,
                                             ndim=3)

        padded_shape = model.padded_shape(shape)

        dest_dataset.add_channel(name=channel,
                                 shape=padded_shape,
                                 dtype=array.dtype,
                                 chunks=chunks,
                                 codec=compression,
                                 clevel=compression_level)

        def process(tp, device):

            try:
                with asection(f"Loading channel: {channel} for time point {tp}"):
                    tp_array = array[tp].compute()

                with CupyBackend(device, exclusive=True):
                    tp_array = model.apply(tp_array, index=tp, pad=True)

                with asection(f"Moving array from backend to numpy."):
                    tp_array = Backend.to_numpy(tp_array, dtype=array.dtype, force_copy=False)

                with asection(f"Saving stabilized stack for time point {tp}, shape:{array.shape}, dtype:{array.dtype}"):
                    dest_dataset.write_stack(channel=channel,
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

        if workers == -1:
            workers = len(devices)
        aprint(f"Number of workers: {workers}")

        if workers > 1:
            Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp, devices[tp % len(devices)]) for tp in range(0, shape[0]))
        else:
            for tp in range(0, shape[0]):
                process(tp, devices[0])
    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
