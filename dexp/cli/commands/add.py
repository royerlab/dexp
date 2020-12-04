import click

from dexp.cli.main import _default_store
from dexp.cli.utils import _get_dataset_from_path, _parse_channels, _get_output_path
from dexp.utils.timeit import timeit


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--rename', '-rc', default=None, help='You can rename channels: e.g. if channels are ‘channel1,anotherc’ then ‘gfp,rfp’ would rename the ‘channel1’ channel to ‘gfp’, and ‘anotherc’ to ‘rfp’ ')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
def add(input_path, output_path, channels, rename, store, overwrite):
    input_dataset = _get_dataset_from_path(input_path)
    output_path = _get_output_path(input_path, output_path)
    channels = _parse_channels(input_dataset, channels)

    if rename is None:
        rename = input_dataset.channels()
    else:
        rename = rename.split(',')

    print(f"New names for channel(s): {rename}")

    with timeit("deconvolution"):
        input_dataset.add_channels_to(output_path,
                                      channels=channels,
                                      rename=rename,
                                      store=store,
                                      overwrite=overwrite
                                      )

    input_dataset.close()
    print("Done!")
