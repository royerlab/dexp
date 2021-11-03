import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import _default_store
from dexp.cli.parsing import _parse_channels, _get_output_path
from dexp.datasets.open_dataset import glob_datasets


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--rename', '-rc', default=None, help='You can rename channels: e.g. if channels are ‘channel1,anotherc’ then ‘gfp,rfp’ would rename the ‘channel1’ channel to ‘gfp’, and ‘anotherc’ to ‘rfp’ ')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
@click.option('--projection', '-p/-np', is_flag=True, default=True, help='If flags should be copied.', show_default=True)
def add(input_paths, output_path, channels, rename, store, overwrite, projection):
    """ Adds the channels selected from INPUT_PATHS to the given output dataset (created if not existing).
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path, '_add')
    channels = _parse_channels(input_dataset, channels)

    if rename is None:
        rename = input_dataset.channels()
    else:
        rename = rename.split(',')

    with asection(f"Adding channels: {channels} from: {input_paths} to {output_path}, with new names: {rename}"):
        input_dataset.add_channels_to(output_path,
                                      channels=channels,
                                      rename=rename,
                                      store=store,
                                      overwrite=overwrite,
                                      add_projections=projection,
                                      )

        input_dataset.close()
        aprint("Done!")
