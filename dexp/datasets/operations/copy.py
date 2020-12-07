import os

from arbol.arbol import aprint, asection


def dataset_copy(dataset,
                 path,
                 channels,
                 slicing,
                 store,
                 compression,
                 compression_level,
                 overwrite,
                 project,
                 workers,
                 check):
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(path, mode, store)

    for channel in dataset._selected_channels(channels):

        array = dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

        if slicing is not None:
            array = array[slicing]

        if project:
            shape = array.shape[0:project] + array.shape[project + 1:]
            dim = len(shape)
            chunks = (1,) + (None,) * (dim - 1)
            aprint(f"projecting along axis {project} to shape: {shape} and chunks: {chunks}")

        else:
            shape = array.shape
            dim = len(shape)
            if dim == 3:
                chunks = dataset._default_chunks[1:]
            elif dim == 4:
                chunks = dataset._default_chunks

        dest_array = dest_dataset.add_channel(name=channel,
                                              shape=shape,
                                              dtype=array.dtype,
                                              chunks=chunks,
                                              codec=compression,
                                              clevel=compression_level)

        def process(tp):
            try:
                with asection(f"Starting to process time point: {tp} ..."):
                    tp_array = array[tp].compute()
                    if project:
                        # project is the axis for projection, but here we are not considering the T dimension anymore...
                        axis = project - 1
                        tp_array = tp_array.max(axis=axis)

                    dest_array[tp] = tp_array
            except Exception as error:
                aprint(error)
                aprint(f"Error occurred while copying time point {tp} !")

        from joblib import Parallel, delayed

        if workers is None:
            workers = os.cpu_count() // 2

        aprint(f"Number of workers: {workers}")
        Parallel(n_jobs=workers)(delayed(process)(tp) for tp in range(0, shape[0]))

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()
    dest_dataset.close()
