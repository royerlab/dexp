from dexp.datasets.operations.crop import dataset_crop
import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import _default_clevel, _default_codec, _default_store
from dexp.cli.parsing import _parse_channels, _get_output_path, _parse_chunks
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.crop import dataset_crop


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--quantile', '-q', default=0.99, type=float, help='Quantile parameter for lower bound of brightness for thresholding.', show_default=True)
@click.option('--reference-channel', '-rc', default=None, help='Reference channel to estimate cropping. If no provided it picks the first one.')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--chunks', '-chk', default=None, help='Dataset chunks dimensions, e.g. (1, 126, 512, 512).')
@click.option('--codec', '-z', default=_default_codec, help='Compression codec: zstd for ’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
@click.option('--workers', '-wk', default=-4, help='Number of worker threads to spawn. Negative numbers n correspond to: number_of _cores / |n| ', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def crop(input_paths,
         output_path,
         channels,
         quantile,
         reference_channel,
         store,
         chunks,
         codec,
         clevel,
         overwrite,
         workers,
         check):

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path, '_crop')
    channels = _parse_channels(input_dataset, channels)
    if reference_channel is None:
        reference_channel = input_dataset.channels()[0]
    chunks = _parse_chunks(chunks)

    with asection(f"Cropping from: {input_paths} to {output_path} for channels: {channels}, "
                  f"using channel {reference_channel} as a reference."):

        dataset_crop(input_dataset,
                     output_path,
                     channels=channels,
                     reference_channel=reference_channel,
                     quantile=quantile,
                     store=store,
                     chunks=chunks,
                     compression=codec,
                     compression_level=clevel,
                     overwrite=overwrite,
                     workers=workers,
                     check=check)

        input_dataset.close()
        aprint("Done!")
