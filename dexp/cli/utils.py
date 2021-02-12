from arbol import aprint
from numpy import s_


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
    aprint(f"Requested slicing    : '{slicing if slicing else '--All--'}' ")
    if slicing is not None:
        aprint(f"Slicing: {slicing}")
        dummy = s_[1, 2]
        slicing = eval(f"s_{slicing}")
    return slicing


def _parse_channels(input_dataset, channels):
    available_channels = frozenset(input_dataset.channels())
    aprint(f"Available channel(s) : '{available_channels}'")
    aprint(f"Requested channel(s) : '{channels if channels else '--All--'}' ")
    if channels is None:
        channels = input_dataset.channels()
    else:
        channels = tuple(channel.strip() for channel in channels.split(','))
        channels = [channel for channel in channels if channel in available_channels]

    aprint(f"Selected channel(s)  : '{channels}'")
    return channels


def _parse_devices(devices):
    aprint(f"Requested devices    :  '{'--All--' if 'all' in devices else devices}' ")
    if 'all' in devices:
        from dexp.processing.backends.cupy_backend import CupyBackend
        devices = list(range(len(CupyBackend.available_devices())))
    else:
        devices = tuple(int(device.strip()) for device in devices.split(','))

    return devices
