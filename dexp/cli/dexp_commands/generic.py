import inspect
from importlib import import_module
from typing import Callable, Optional, Sequence, Union

import click
from arbol import asection
from toolz import curry

from dexp.cli.parsing import (
    args_option,
    channels_option,
    input_dataset_argument,
    multi_devices_option,
    output_dataset_options,
    parse_args_to_kwargs,
    slicing_option,
    tilesize_option,
    validate_function_kwargs,
)
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.operations.generic import dataset_generic
from dexp.datasets.zarr_dataset import ZDataset


def _parse_function(func_name: str, pkg_import: str, args: Sequence[str]) -> Callable:
    pkg = import_module(pkg_import)
    if not hasattr(pkg, func_name):
        functions = inspect.getmembers(pkg, inspect.isfunction)
        functions = [f[0] for f in functions]
        raise ValueError(f"Could not find function {func_name} in {pkg_import}\n. Available functions {functions}")

    kwargs = parse_args_to_kwargs(args)
    func = getattr(pkg, func_name)
    validate_function_kwargs(func, kwargs)

    func = curry(func, **kwargs)
    return func


@click.command(context_settings=dict(ignore_unknown_options=True))
@input_dataset_argument()
@output_dataset_options()
@channels_option()
@multi_devices_option()
@slicing_option()
@tilesize_option(default=None)
@args_option()
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
