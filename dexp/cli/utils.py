from os.path import exists, join

from numpy import s_

from dexp.datasets.clearcontrol_dataset import CCDataset
from dexp.datasets.zarr_dataset import ZDataset


def _get_dataset_from_path(input_path):
    if exists(join(input_path, 'stacks')):
        input_dataset = CCDataset(input_path)
    else:
        input_dataset = ZDataset(input_path)
    return input_dataset


def _get_output_path(input_path, output_path, postfix=''):
    if output_path is None or not output_path.strip():
        if input_path.endswith('/') or input_path.endswith('\\'):
            input_path = input_path[:-1]
        if input_path.endswith('.zip'):
            input_path = input_path[:-4]
        if input_path.endswith('.nested.zarr'):
            input_path = input_path[:-12]
        if input_path.endswith('.zarr'):
            input_path = input_path[:-5]
        return input_path + postfix
    else:
        return output_path


def _parse_slicing(slicing: str):
    print(f"Requested slicing    : '{slicing if slicing else '--All--'}' ")
    if slicing is not None:
        print(f"Slicing: {slicing}")
        dummy = s_[1, 2]
        slicing = eval(f"s_{slicing}")
    return slicing


def _parse_channels(input_dataset, channels):
    print(f"Available channel(s) : '{input_dataset.channels()}'")
    print(f"Requested channel(s) : '{channels if channels else '--All--'}' ")
    if channels is None:
        channels = input_dataset.channels()
    else:
        channels = tuple(channel.strip() for channel in channels.split(','))
        channels = list(set(channels) & set(input_dataset.channels()))
    print(f"Selected channel(s)  : '{channels}'")
    return channels


def _parse_devices(devices):
    print(f"Requested devices    :  '{devices if devices else '--All--'}' ")
    if devices is not None:
        devices = tuple(device.strip() for device in devices.split(','))
    return devices
