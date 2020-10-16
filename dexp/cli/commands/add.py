from time import time

import click

from dexp.cli.main import _get_dataset_from_path, _default_store


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when ommited.')
@click.option('--rename', '-rc', default=None, help='You can rename channels: e.g. if channels are `channel1,anotherc` then `gfp,rfp` would rename the `channel1` channel to `gfp`, and `anotherc` to `rfp` ')
@click.option('--store', '-st', default=_default_store, help='Store: ‘dir’, ‘zip’', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
def add(input_path, output_path, channels, rename, store, overwrite):
    input_dataset = _get_dataset_from_path(input_path)

    if channels is None:
        selected_channels = input_dataset.channels()
    else:
        channels = channels.split(',')
        selected_channels = list(set(channels) & set(input_dataset.channels()))

    if rename is None:
        rename = input_dataset.channels()
    else:
        rename = rename.split(',')

    print(f"Available channel(s)    : {input_dataset.channels()}")
    print(f"Requested channel(s)    : {channels}")
    print(f"Selected channel(s)     : {selected_channels}")
    print(f"New names for channel(s): {rename}")

    time_start = time()
    input_dataset.add_channels_to(output_path,
                                  channels=selected_channels,
                                  rename=rename,
                                  store=store,
                                  overwrite=overwrite,
                                  )
    time_stop = time()
    print(f"Elapsed time: {time_stop - time_start} seconds")
    input_dataset.close()
    print("Done!")
