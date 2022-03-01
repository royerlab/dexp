from typing import Optional, Sequence, Tuple, Union

from arbol import aprint
from numpy import s_


def _get_output_path(input_path, output_path, postfix=""):
    if output_path is None or not output_path.strip():
        return _strip_path(input_path) + postfix
    elif output_path.startswith("_"):
        return _strip_path(input_path) + postfix + output_path
    else:
        return output_path


def _strip_path(path):
    if path.endswith("/") or path.endswith("\\"):
        path = path[:-1]
    if path.endswith(".zip"):
        path = path[:-4]
    if path.endswith(".nested.zarr"):
        path = path[:-12]
    if path.endswith(".zarr"):
        path = path[:-5]
    return path


def _parse_slicing(slicing: str):
    aprint(f"Requested slicing    : '{slicing if slicing else '--All--'}' ")
    if slicing is not None:
        aprint(f"Slicing: {slicing}")
        _ = s_[1, 2]  # ??
        slicing = eval(f"s_{slicing}")
    return slicing


def _parse_channels(input_dataset, channels):
    available_channels = frozenset(input_dataset.channels())
    aprint(f"Available channel(s) : '{available_channels}'")
    aprint(f"Requested channel(s) : '{channels if channels else '--All--'}' ")
    if channels is None:
        channels = input_dataset.channels()
    else:
        channels = tuple(channel.strip() for channel in channels.split(","))
        channels = [channel for channel in channels if channel in available_channels]

    aprint(f"Selected channel(s)  : '{channels}'")
    return channels


def parse_devices(devices: str) -> Union[str, Sequence[int]]:
    aprint(f"Requested devices    :  '{'--All--' if 'all' in devices else devices}' ")

    if devices.endswith(".json"):
        return devices

    elif "all" in devices:
        from dexp.utils.backends import CupyBackend

        devices = list(range(len(CupyBackend.available_devices())))

    else:
        devices = list(int(device.strip()) for device in devices.split(","))

    return devices


def _parse_chunks(chunks: Optional[str]) -> Optional[Tuple[int]]:
    if chunks is not None:
        chunks = eval(chunks)
    return chunks
