from typing import Optional, Sequence

import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import (
    channels_option,
    input_dataset_argument,
    multi_devices_option,
    output_dataset_options,
    slicing_option,
)
from dexp.datasets.operations.deskew import dataset_deskew
from dexp.datasets.zarr_dataset import ZDataset


def _flip_callback(ctx: click.Context, opt: click.Option, value: Optional[str]) -> Sequence[str]:
    length = len(ctx.params["input_dataset"].channels())
    if value is None:
        value = (False,) * length
    else:
        value = tuple(bool(s) for s in value.split(","))
    if len(value) != length:
        raise ValueError("-fl --flips must have length equal to number of channels.")
    return value


@click.command()
@input_dataset_argument()
@output_dataset_options()
@slicing_option()
@channels_option()
@multi_devices_option()
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["yang", "classic"]),
    default="yang",
    help="Deskew algorithm",
    show_default=True,
)
@click.option("--delta-x", "-xx", type=float, default=None, help="Pixel size of the camera", show_default=True)
@click.option(
    "--delta-z",
    "-zz",
    type=float,
    default=None,
    help="Scanning step (stage or galvo scanning step, not the same as the distance between the slices)",
    show_default=True,
)
@click.option(
    "--angle",
    "-a",
    type=float,
    default=None,
    help="Incident angle of the light sheet, angle between the light sheet and the optical axis in degrees",
    show_default=True,
)  #
@click.option(
    "--flips",
    "-fl",
    type=str,
    default=None,
    help="Flips image to deskew in the opposite orientation (True for view 0 and False for view 1)",
    show_default=True,
    callback=_flip_callback,
)
@click.option(
    "--camera-orientation",
    "-co",
    type=int,
    default=0,
    help="Camera orientation correction expressed as a number of 90 deg rotations to be performed per 2D image in stack -- if required.",
    show_default=True,
)
@click.option("--depth-axis", "-za", type=int, default=0, help="Depth axis.", show_default=True)
@click.option("--lateral-axis", "-xa", type=int, default=1, help="Lateral axis.", show_default=True)
@click.option(
    "--padding", "-p", type=bool, default=False, is_flag=True, help="Pads output image to fit deskwed results."
)
def deskew(
    input_dataset: ZDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    mode: str,
    delta_x: float,
    delta_z: float,
    angle: float,
    flips: Sequence[bool],
    camera_orientation: int,
    depth_axis: int,
    lateral_axis: int,
    padding: bool,
    devices: Sequence[int],
):
    """Deskews all or selected channels of a dataset."""
    with asection(
        f"Deskewing dataset: {input_dataset.path}, saving it at: {output_dataset.path}, for channels: {channels}"
    ):
        aprint(f"Devices used: {devices}")
        dataset_deskew(
            input_dataset,
            output_dataset,
            channels=channels,
            dx=delta_x,
            dz=delta_z,
            angle=angle,
            flips=flips,
            camera_orientation=camera_orientation,
            depth_axis=depth_axis,
            lateral_axis=lateral_axis,
            mode=mode,
            padding=padding,
            devices=devices,
        )

    input_dataset.close()
    output_dataset.close()
