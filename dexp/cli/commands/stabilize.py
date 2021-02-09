import click
from arbol.arbol import aprint, asection

from dexp.cli.dexp_main import _default_store, _default_codec, _default_clevel
from dexp.cli.dexp_main import _default_workers_backend
from dexp.cli.utils import _parse_channels, _get_dataset_from_path, _get_output_path, _parse_slicing, _parse_devices
from dexp.datasets.operations.stabilize import dataset_stabilize


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)
@click.option('--chunksize', '-cs', type=int, default=512, help='Chunk size for tiled computation', show_default=True)
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
@click.option('--devices', '-d', type=str, default='0', help='Sets the CUDA devices id, e.g. 0,1,2 or ‘all’', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def deconv(input_path,
           output_path,
           channels,
           slicing,
           store,
           codec,
           clevel,
           overwrite,
           chunksize,
           workers,
           workersbackend,
           devices,
           check):
    input_dataset = _get_dataset_from_path(input_path)
    output_path = _get_output_path(input_path, output_path, "_stabilized")

    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)
    devices = _parse_devices(devices)

    with asection(f"Stabilizing dataset: {input_path}, saving it at: {output_path}, for channels: {channels}, slicing: {slicing} "):
        dataset_stabilize(input_dataset,
                          output_path,
                          channels=channels,
                          slicing=slicing,
                          store=store,
                          compression=codec,
                          compression_level=clevel,
                          overwrite=overwrite,
                          workers=workers,
                          workersbackend=workersbackend,
                          devices=devices,
                          check=check
                          )

        input_dataset.close()
        aprint("Done!")
