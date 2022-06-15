from typing import Optional, Sequence, Union

import click
from arbol import asection

from dexp.cli.parsing import (
    KwargsCommand,
    args_option,
    channels_option,
    func_args_to_str,
    input_dataset_argument,
    multi_devices_option,
    output_dataset_options,
    parse_args_to_kwargs,
    slicing_option,
    validate_function_kwargs,
)
from dexp.datasets import BaseDataset, ZDataset
from dexp.datasets.operations.background import dataset_background
from dexp.processing.crop.background import foreground_mask


@click.command(
    cls=KwargsCommand,
    context_settings=dict(ignore_unknown_options=True),
    epilog=func_args_to_str(foreground_mask, ["array"]),
)
@input_dataset_argument()
@output_dataset_options()
@channels_option()
@multi_devices_option()
@slicing_option()
@args_option()
@click.option(
    "--reference-channel", "-rc", type=str, default=None, help="Optional channel to use for background removal."
)
@click.option(
    "--merge-channels",
    "-mc",
    type=bool,
    is_flag=True,
    default=False,
    help="Flag to indicate to merge (with addition) the image channels to detect the foreground.",
)
def background(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    reference_channel: Optional[str],
    merge_channels: bool,
    devices: Union[str, Sequence[int]],
    args: Sequence[str],
) -> None:
    """Remove background by detecting foreground regions using morphological analysis."""
    kwargs = parse_args_to_kwargs(args)
    validate_function_kwargs(foreground_mask, kwargs)

    with asection(f"Removing background of {input_dataset.path}."):
        dataset_background(
            input_dataset=input_dataset,
            output_dataset=output_dataset,
            channels=channels,
            reference_channel=reference_channel,
            merge_channels=merge_channels,
            devices=devices,
            **kwargs,
        )

    input_dataset.close()
    output_dataset.close()
