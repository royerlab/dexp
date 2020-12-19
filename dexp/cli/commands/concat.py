import click
from arbol.arbol import aprint, asection
from joblib import Parallel, delayed

from dexp.cli.main import _default_store, _default_codec, _default_clevel, _default_workers_backend
from dexp.cli.utils import _get_dataset_from_path, _parse_channels, _get_output_path


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--rename', '-rc', default=None, help='You can rename channels: e.g. if channels are ‘channel1,anotherc’ then ‘gfp,rfp’ would rename the ‘channel1’ channel to ‘gfp’, and ‘anotherc’ to ‘rfp’ ')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ')
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, if -1 then num workers = num channels', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
def concat(input_paths, output_path, channels, store, codec, clevel, overwrite, workers, workersbackend):

    input_datasets= tuple(_get_dataset_from_path(input_path) for input_path in input_paths)
    output_path = _get_output_path(input_paths[0], output_path, '_concat')
    channels = _parse_channels(input_datasets[0], channels)

    with asection(f"Concatenting channels: {channels} from: "):
        for input_path, input_dataset in zip(input_paths, input_datasets):
            aprint(f"dataset at path: {input_path} of shape: {input_dataset.shape()}")

    shapes = tuple(input_dataset.shape() for input_dataset in input_datasets)
    dtypes = tuple(input_dataset.dtype() for input_dataset in input_datasets)

    for shape, dtype in zip(shapes, dtypes):
        if shape[0:-1] != shapes[0] or dtype != dtypes[0]:
            aprint("Error: can't concatenate arrays of different shape!")
            return

    total_num_timepoints = sum(shape[0] for shape in shapes)
    shape = (total_num_timepoints,)+shapes[0][1:]
    dtype = dtypes[0]

    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(output_path, mode, store)

    def process(channel):
        with asection(f"Processing channel: {channel}"):
            dest_dataset.add_channel(name=channel,
                                     shape=shape,
                                     dtype=dtype,
                                     codec=codec,
                                     clevel=clevel)

            new_array = dest_dataset.get_array(channel, per_z_slice=False)

            start = 0
            for i, dataset in enumerate(input_datasets):
                array = dataset.get_array(channel, per_z_slice=False)
                num_timepoints = array.shape[0]
                aprint(f"Adding timepoints: [{start}, {start+num_timepoints}] from dataset #{i} ")
                new_array[start, start+num_timepoints] = array
                start += num_timepoints

    if workers == -1:
        workers = len(channels)
    aprint(f"Number of workers: {workers}")

    if workers > 1:
        Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(channel) for channel in channels)
    else:
        for channel in channels:
            process(channel)

    dest_dataset.close()
    input_dataset.close()
    aprint("Done!")
