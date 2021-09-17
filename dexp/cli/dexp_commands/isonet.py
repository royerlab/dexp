import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import _default_clevel, _default_codec, _default_store
from dexp.cli.parsing import _get_output_path, _parse_slicing
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.isonet import dataset_isonet


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--dxy', '-dxy', type=float, required=True)
@click.option('--dz', '-dz', type=float, required=True)
@click.option('--binning', '-b', type=int, default=1)  # TODO help
@click.option('--sharpening', '-sh', is_flag=True, help='Dehazes the image with a low-pass rejection filter.')
@click.option('--channel', '-ch', default=None, help='dataset channel name')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)  #
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--context', '-c', default='default', help="IsoNet context name", show_default=True)  # , help='dataset slice'
@click.option('--mode', '-m', default='pta', help="mode: 'pta' -> prepare, train and apply, 'a' just apply  ", show_default=True)  # , help='dataset slice'
@click.option('--max_epochs', '-e', type=int, default='50', help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def isonet(input_paths,
           output_path,
           dxy,
           dz,
           sharpening,
           binning,
           channel,
           slicing,
           store,
           codec,
           clevel,
           overwrite,
           context,
           mode,
           max_epochs,
           check):
    """ Recovers isotropic resolution using the ISONET approach (Weigert et al.)
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path, '_isonet')
    slicing = _parse_slicing(slicing)

    with asection(f"Applying Isonet to: {input_paths}, saving result to: {output_path}, slicing: {slicing} "):
        dataset_isonet(input_dataset,
                       output_path,
                       dxy=dxy,
                       dz=dz,
                       channel=channel,
                       slicing=slicing,
                       store=store,
                       compression=codec,
                       compression_level=clevel,
                       overwrite=overwrite,
                       context=context,
                       mode=mode,
                       max_epochs=max_epochs,
                       check=check,
                       sharpening=sharpening,
                       binning=binning,
                       training_tp_index=None)

        input_dataset.close()
        aprint("Done!")
