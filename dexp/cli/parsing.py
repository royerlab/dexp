import ast
import inspect
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import click
from arbol import aprint
from numpy import s_
from toolz import curry

from dexp.cli.defaults import DEFAULT_CLEVEL, DEFAULT_CODEC, DEFAULT_STORE
from dexp.datasets import ZDataset
from dexp.datasets.open_dataset import glob_datasets
from dexp.utils import overwrite2mode


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


def multi_devices_option() -> Callable:
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


def device_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--device",
            "-d",
            type=int,
            default=0,
            show_default=True,
            help="Selects the CUDA device by id, starting from 0.",
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


def empty_channels_callback(ctx: click.Context, opt: click.Option, value: Optional[str]) -> Sequence[str]:
    """Returns empty list if no channels are provided."""
    if value is None:
        return []
    return _parse_channels(ctx.params["input_dataset"], value)


def input_dataset_callback(ctx: click.Context, arg: click.Argument, value: Sequence[str]) -> None:
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
    mode = overwrite2mode(ctx.params.pop("overwrite"))
    if value is None:
        # new name with suffix if value is None
        value = _get_output_path(ctx.params["input_dataset"].path, None, "." + ctx.command.name)

    ctx.params["output_dataset"] = ZDataset(
        value,
        mode=mode,
        store=ctx.params.pop("store"),
        codec=ctx.params.pop("codec"),
        clevel=ctx.params.pop("clevel"),
        chunks=_parse_chunks(ctx.params.pop("chunks")),
        parent=ctx.params["input_dataset"],
    )
    # removing used parameters
    opt.expose_value = False


def output_dataset_options() -> Callable:
    click_options = [
        click.option(
            "--output-path", "-o", default=None, help="Dataset output path.", callback=output_dataset_callback
        ),
        overwrite_option(),
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


def overwrite_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--overwrite",
            "-w",
            is_flag=True,
            help="Forces overwrite of output",
            show_default=True,
            default=False,
            is_eager=True,
        )(f)

    return decorator


def tilesize_option(default: Optional[int] = 320) -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--tilesize", "-ts", type=int, default=default, help="Tile size for tiled computation", show_default=True
        )(f)

    return decorator


def args_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--args",
            "-a",
            type=str,
            required=False,
            multiple=True,
            default=list(),
            help="Function arguments, it must be used multiple times for multiple arguments. Example: -a sigma=1,2,3 -a pad=constant",
        )(f)

    return decorator


def verbose_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--verbose", "-v", type=bool, is_flag=True, default=False, help="Flag to display intermediated results."
        )(f)

    return decorator


class KwargsCommand(click.Command):
    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Writes the epilog into the formatter if it exists."""
        if self.epilog:
            formatter.write_paragraph()
            with formatter.indentation():
                formatter.write(self.epilog)


def parse_args_to_kwargs(args: Sequence[str], sep: str = "=") -> Dict[str, Any]:
    """Parses list of strings with keys and values to a dictionary,
        given a separator to split the key and values.

    Parameters
    ----------
    args : Sequence[str]
        List of strings of key/values, for example ["sigma=1,2,3", "pad=constant"]
    sep : str, optional
        key-value separator, by default "="

    Returns
    -------
    Dict[str, Any]
        Dictionary of key word arguments.
    """
    args = [arg.split(sep) for arg in args]
    assert all(len(arg) == 2 for arg in args)

    kwargs = dict(args)
    kwargs = {k: ast.literal_eval(v) for k, v in kwargs.items()}

    aprint(f"Using key word arguments {kwargs}")

    return kwargs


def validate_function_kwargs(func: Callable, kwargs: Dict[str, Any]) -> None:
    """Checks if key-word arguments are valid arguments to the provided function.

    Parameters
    ----------
    func : Callable
        Reference function.
    kwargs : Dict[str, Any]
        Key-word arguments as dict.

    Raises
    ------
    ValueError
        If any key isn't an argument of the given function.
    """
    sig = inspect.signature(func)
    for key in kwargs:
        if key not in sig.parameters:
            raise ValueError(f"Function {func.__name__} doesn't support keyword {key}")


def func_args_to_str(func: Callable, ignore: Sequence[str] = []) -> str:
    """Parses functions arguments to string format for click.Command.epilog

    Usage example:
    @click.command(
        cls=KwargsCommand,
        epilog=func_args_to_str(<your function>, <ignored arguments>),
    )
    def func(args: Sequence[str]):
        pass

    Parameters
    ----------
    func : Callable
        Function to extract arguments and default values
    ignore : Sequence[str], optional
        Arguments to ignore, by default []

    Returns
    -------
    str
        String for click.command(epilog)
    """
    sig = inspect.signature(func)
    text = "Known key-word arguments:\n"
    for k, v in sig.parameters.items():
        if k not in ignore:
            text += f"  -a, --args {k}={v.default}\n"

    return text


@curry
def tuple_callback(
    ctx: click.Context,
    opt: click.Option,
    value: str,
    dtype: Callable = int,
    length: Optional[int] = None,
) -> Optional[Tuple[Any]]:
    """Parses string to tuple given dtype and optional length.
       Returns None if None is supplied.

    Parameters
    ----------
    ctx : click.Context
        CLI context, not used.
    opt : click.Option
        CLI option, not used.
    value : str
        Input value.
    dtype : Callable, optional
        Data type for type casting, by default int
    length : Optional[int], optional
        Optional length for length checking, by default None

    Returns
    -------
    Tuple[Any]
        Tuple of given dtype and length (optional).
    """
    if value is None:
        return None
    tup = tuple(dtype(s) for s in value.split(","))
    if length is not None and length != len(tup):
        raise ValueError(f"Expected tuple of length {length}, got input {tup}")
    return tup
