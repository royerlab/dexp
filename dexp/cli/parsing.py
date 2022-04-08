import warnings
from typing import Callable, Optional, Sequence, Tuple, Union

import click
from arbol import aprint
from numpy import s_

from dexp.cli.defaults import DEFAULT_CLEVEL, DEFAULT_CODEC, DEFAULT_STORE
from dexp.datasets import ZDataset
from dexp.datasets.open_dataset import glob_datasets


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
        chunks = (int(c) for c in chunks.split(","))
    return chunks


def devices_callback(ctx: click.Context, opt: click.Option, value: str) -> Sequence[int]:
    return parse_devices(value)


def devices_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--devices",
            "-d",
            type=str,
            default="all",
            help="Sets the CUDA devices id, e.g. 0,1,2 or ‘all’",
            show_default=True,
            callback=devices_callback,
        )(f)

    return decorator


def workers_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--workers",
            "-wk",
            default=-4,
            help="Number of worker threads to spawn. Negative numbers n correspond to: number_of _cores / |n| ",
            show_default=True,
        )(f)

    return decorator


def slicing_callback(ctx: click.Context, opt: click.Option, value: str) -> None:
    slicing = _parse_slicing(value)
    if slicing is not None:
        ctx.params["input_dataset"].set_slicing(slicing)
    opt.expose_value = False


def slicing_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--slicing",
            "-s",
            default=None,
            help="dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ",
            callback=slicing_callback,
        )(f)

    return decorator


def channels_callback(ctx: click.Context, opt: click.Option, value: Optional[str]) -> Sequence[str]:
    return _parse_channels(ctx.params["input_dataset"], value)


def channels_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--channels",
            "-c",
            default=None,
            help="list of channels, all channels when ommited.",
            callback=channels_callback,
        )(f)

    return decorator


def optional_channels_callback(ctx: click.Context, opt: click.Option, value: Optional[str]) -> Sequence[str]:
    """Parses optional channels name, returns `input_dataset` channels if nothing is provided."""
    return ctx.params["input_dataset"].channels() if value is None else value.split(",")


def input_dataset_callback(ctx: click.Context, arg: click.Argument, value: str) -> str:
    try:
        ctx.params["input_dataset"], _ = glob_datasets(value)
        arg.expose_value = False

    except ValueError as e:
        warnings.warn(str(e))


def input_dataset_argument() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.argument("input-paths", nargs=-1, callback=input_dataset_callback, is_eager=True)(f)

    return decorator


def output_dataset_callback(ctx: click.Context, opt: click.Option, value: Optional[str]) -> None:
    mode = "w" if ctx.params["overwrite"] else "w-"
    if value is None:
        # new name with suffix if value is None
        value = _get_output_path(ctx.params["input_dataset"].path, None, "." + ctx.command.name)

    ctx.params["output_dataset"] = ZDataset(
        value,
        mode=mode,
        store=ctx.params["store"],
        codec=ctx.params["codec"],
        clevel=ctx.params["clevel"],
        chunks=_parse_chunks(ctx.params["chunks"]),
        parent=ctx.params["input_dataset"],
    )
    # removing used parameters
    opt.expose_value = False
    for key in ["overwrite", "store", "codec", "clevel", "chunks"]:
        del ctx.params[key]


def output_dataset_options() -> Callable:

    click_options = [
        click.option(
            "--output-path", "-o", default=None, help="Dataset output path.", callback=output_dataset_callback
        ),
        click.option(
            "--overwrite",
            "-w",
            is_flag=True,
            help="Forces overwrite of target",
            show_default=True,
            default=False,
            is_eager=True,
        ),
        click.option(
            "--store",
            "-st",
            default=DEFAULT_STORE,
            help="Zarr store: ‘dir’, ‘ndir’, or ‘zip’",
            show_default=True,
            is_eager=True,
        ),
        click.option(
            "--chunks", "-chk", default=None, help="Dataset chunks dimensions, e.g. (1, 126, 512, 512).", is_eager=True
        ),
        click.option(
            "--codec",
            "-z",
            default=DEFAULT_CODEC,
            help="Compression codec: ‘zstd‘, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’",
            show_default=True,
            is_eager=True,
        ),
        click.option(
            "--clevel",
            "-l",
            type=int,
            default=DEFAULT_CLEVEL,
            help="Compression level",
            show_default=True,
            is_eager=True,
        ),
    ]

    def decorator(f: Callable) -> Callable:
        for opt in click_options:
            f = opt(f)
        return f

    return decorator
