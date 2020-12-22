import os

from arbol.arbol import aprint, asection
from joblib import Parallel, delayed


def dataset_concat(channels,
                   input_datasets,
                   output_path,
                   overwrite,
                   store,
                   codec,
                   clevel,
                   workers,
                   workersbackend):
    # Create destination dataset:
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(output_path, mode, store)

    # Per time point processing:
    def process(channel):
        with asection(f"Processing channel: {channel}"):

            # collecting shapes and dtypes:
            shapes = tuple(input_dataset.shape(channel) for input_dataset in input_datasets)
            dtypes = tuple(input_dataset.dtype(channel) for input_dataset in input_datasets)
            aprint(f"shapes: {shapes}")
            aprint(f"dtypes: {dtypes}")
            for shape, dtype in zip(shapes, dtypes):
                if shape[0:-1] != shapes[0][0:-1] or dtype != dtypes[0]:
                    raise ValueError("Error: can't concatenate arrays of different shape!")

            # deciding of shape and dtype of concatenateed array:
            total_num_timepoints = sum(shape[0] for shape in shapes)
            shape = (total_num_timepoints,) + shapes[0][1:]
            dtype = dtypes[0]
            aprint(f"Adding channel: {channel} of shape: {shape} and dtype: {dtype} with codec:{codec}, and clevel: {clevel} to concatenated dataset.")

            # We add each channel in the concatenated dataset:
            dest_dataset.add_channel(name=channel,
                                     shape=shape,
                                     dtype=dtype,
                                     codec=codec,
                                     clevel=clevel)

            # get the array:
            new_array = dest_dataset.get_array(channel, per_z_slice=False)

            # We add copy from the input arrays:
            start = 0
            for i, dataset in enumerate(input_datasets):
                array = dataset.get_array(channel, per_z_slice=False)
                num_timepoints = array.shape[0]
                aprint(f"Adding timepoints: [{start}, {start + num_timepoints}] from dataset #{i} ")
                new_array[start:start + num_timepoints] = array
                start += num_timepoints

    # Workers:
    if workers == -1:
        workers = min(len(channels), os.cpu_count() // 2)
    aprint(f"Number of workers: {workers}")
    if workers > 1:
        Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(channel) for channel in channels)
    else:
        for channel in channels:
            process(channel)

    # close destination dataset:
    dest_dataset.close()
