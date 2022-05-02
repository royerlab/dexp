import inspect
from importlib import import_module
from typing import Callable, Optional, Sequence, Union

import click
from arbol import aprint, asection
from toolz import curry

from dexp.cli.parsing import (
    channels_option,
    input_dataset_argument,
    multi_devices_option,
    output_dataset_options,
    slicing_option,
    tilesize_option,
)
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.operations.generic import dataset_generic
from dexp.datasets.zarr_dataset import ZDataset


def _parse_function(func_name: str, pkg_import: str, args: Sequence[str]) -> Callable:
    args = [arg.split("=") for arg in args]
    assert all(len(arg) == 2 for arg in args)

    kwargs = dict(args)
    aprint(f"Using key word arguments {kwargs}")

    pkg = import_module(pkg_import)
    if not hasattr(pkg, func_name):
        raise ValueError(
            f"Could not find function {func_name} in {pkg_import}\n. Available attributes {inspect.getmembers(pkg, inspect.isfunction)}"
        )

    func = getattr(pkg, func_name)
    sig = inspect.signature(func)
    for key in kwargs:
        if key not in sig.parameters:
            raise ValueError(f"Function {func.__name__} doesn't support keyword {key}")

    func = curry(func, **kwargs)
    return func


@click.command(context_settings=dict(ignore_unknown_options=True))
@input_dataset_argument()
@output_dataset_options()
@channels_option()
@multi_devices_option()
@slicing_option()
@tilesize_option(default=None)
@click.option(
    "--func-name",
    "-f",
    type=str,
    required=True,
    help="Function to be called, arguments should be provided as keyword arguments on the cli.",
)
@click.option(
    "--pkg-import",
    "-pkg",
    type=str,
    required=True,
    help="Package to be imported for function usage. For example, `cupyx.scipy.ndimage` when using `gaussian_filter`.",
)
@click.option(
    "--args",
    "-a",
    type=str,
    required=False,
    multiple=True,
    default=list(),
    help="Function arguments, it must be used multiple times for multiple arguments.",
)
def generic(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    func_name: str,
    pkg_import: str,
    tilesize: Optional[int],
    devices: Union[str, Sequence[int]],
    args: Sequence[str],
):
    """
    Executes a generic function in multiples gpus using dask and cupy.
    """
    func = _parse_function(func_name=func_name, pkg_import=pkg_import, args=args)
    with asection(f"Applying {func.__name__} to {output_dataset.path} for channels {channels}"):
        dataset_generic(
            input_dataset,
            output_dataset,
            func=func,
            channels=channels,
            tilesize=tilesize,
            devices=devices,
        )

    input_dataset.close()
    output_dataset.close()
