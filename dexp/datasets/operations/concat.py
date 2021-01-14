import os
from typing import Tuple, Sequence

from arbol.arbol import aprint, asection
from joblib import Parallel, delayed

from dexp.datasets.base_dataset import BaseDataset


def dataset_concat(channels: Sequence[str],
                   input_datasets: Tuple[BaseDataset],
                   output_path: str,
                   overwrite: bool,
                   store: str,
                   codec: str,
                   clevel: int,
                   workers: int,
                   workersbackend: str):
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
                if shape[1:] != shapes[0][1:] or dtype != dtypes[0]:
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

            # get the destination array:
            new_array = dest_dataset.get_array(channel, per_z_slice=False)
            ndim = new_array.ndim - 1

            # get destination projection arrays:
            new_proj_arrays = tuple(dest_dataset.get_projection_array(channel, axis) for axis in range(ndim))

            # We add copy from the input arrays:
            start = 0
            for i, dataset in enumerate(input_datasets):
                num_timepoints = dataset.shape(channel)[0]
                aprint(f"Adding timepoints: [{start}, {start + num_timepoints}] from dataset #{i} ")

                try:
                    # adding projections:
                    for axis in range(ndim):
                        proj_array = dataset.get_projection_array(channel, axis)
                        new_proj_arrays[axis][start:start + num_timepoints] = proj_array

                    # adding main data:
                    array = dataset.get_array(channel, per_z_slice=False)
                    new_array[start:start + num_timepoints] = array
                except KeyError:
                    aprint("Projections missing for ")
                    # this happens if we don't have projections, in that case we need to generate the projections:
                    # slower but necessary...
                    for tp_src in range(num_timepoints):
                        tp_dest = start+tp_src
                        dest_dataset.write_stack(channel, tp_dest, array[tp_src])

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
