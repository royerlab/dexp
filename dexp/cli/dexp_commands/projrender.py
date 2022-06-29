from pathlib import Path
from typing import Optional, Sequence, Union

import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import (
    channels_option,
    input_dataset_argument,
    multi_devices_option,
    overwrite_option,
    slicing_option,
    tuple_callback,
)
from dexp.datasets.operations.projrender import dataset_projection_rendering
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@input_dataset_argument()
@slicing_option()
@channels_option()
@multi_devices_option()
@overwrite_option()
@click.option(
    "--output_path",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("frames"),
    help="Output folder to store rendered PNGs. Default is: frames/<channel_name>",
)
@click.option(
    "--axis", "-ax", type=int, default=0, help="Sets the projection axis: 0->Z, 1->Y, 2->X ", show_default=True
)
@click.option(
    "--dir",
    "-di",
    type=int,
    default=-1,
    help="Sets the projection direction: -1 -> top to bottom, +1 -> bottom to top.",
    show_default=True,
)  # , help='dataset slice'
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["max", "maxcolor", "colormax"]),
    default="colormax",
    help="Sets the projection mode: ‘max’: classic max projection, ‘colormax’: color max projection, i.e. color codes for depth, ‘maxcolor’ same as colormax but first does depth-coding by color and then max projects (acheives some level of transparency). ",
    show_default=True,
)
@click.option(
    "--clim",
    "-cl",
    type=str,
    default=None,
    help="Sets the contrast limits, i.e. -cl 0,1000 sets the contrast limits to [0,1000]",
    callback=tuple_callback(dtype=float, length=2),
)
@click.option(
    "--attenuation",
    "-at",
    type=float,
    default=0.1,
    help="Sets the projection attenuation coefficient, should be within [0, 1] ideally close to 0. Larger values mean more attenuation.",
    show_default=True,
)
@click.option(
    "--gamma",
    "-g",
    type=float,
    default=1.0,
    help="Sets the gamma coefficient pre-applied to the raw voxel values (before projection or any subsequent processing).",
    show_default=True,
)
@click.option(
    "--dlim",
    "-dl",
    type=str,
    default=None,
    help="Sets the depth limits. Depth limits. For example, a value of (0.1, 0.7) means that the colormap start at a normalised depth of 0.1, and ends at a normalised depth of 0.7, other values are clipped. Only used for colormax mode.",
    show_default=True,
    callback=tuple_callback(dtype=float, length=2),
)
@click.option(
    "--colormap",
    "-cm",
    type=str,
    default=None,
    help="sets colormap, e.g. viridis, gray, magma, plasma, inferno. Use a rainbow colormap such as turbo, bmy, or rainbow (recommended) for color-coded depth modes. ",
    show_default=True,
)
@click.option(
    "--rgbgamma",
    "-cg",
    type=float,
    default=1.0,
    help="Gamma correction applied to the resulting RGB image. Usefull to brighten image",
    show_default=True,
)
@click.option(
    "--transparency",
    "-t",
    is_flag=True,
    help="Enables transparency output when possible. Good for rendering on white (e.g. on paper).",
    show_default=True,
)
@click.option(
    "--legend-size",
    "-lsi",
    type=float,
    default=1.0,
    help="Multiplicative factor to control size of legend. If 0, no legend is generated.",
    show_default=True,
)
@click.option(
    "--legend-scale",
    "-lsc",
    type=float,
    default=1.0,
    help="Float that gives the scale in some unit of each voxel (along the projection direction). Only in color projection modes.",
    show_default=True,
)
@click.option(
    "--legend-title",
    "-lt",
    type=str,
    default="color-coded depth (voxels)",
    help="Title for the color-coded depth legend.",
    show_default=True,
)
@click.option(
    "--legend-title-color",
    "-ltc",
    type=str,
    default="1,1,1,1",
    help="Legend title color as a tuple of normalised floats: R, G, B, A  (values between 0 and 1).",
    show_default=True,
    callback=tuple_callback(dtype=float, length=4),
)
@click.option(
    "--legend-position",
    "-lp",
    type=str,
    default="bottom_left",
    help="Position of the legend in pixels in natural order: x,y. Can also be a string: bottom_left, bottom_right, top_left, or top_right.",
    show_default=True,
)
@click.option(
    "--legend-alpha",
    "-la",
    type=float,
    default=1,
    help="Transparency for legend (1 means opaque, 0 means completely transparent)",
    show_default=True,
)
def projrender(
    input_dataset: ZDataset,
    output_path: Path,
    channels: Sequence[str],
    overwrite: bool,
    axis: int,
    dir: int,
    mode: str,
    clim: Optional[Sequence[float]],
    attenuation: float,
    gamma: float,
    dlim: Optional[Sequence[float]],
    colormap: str,
    rgbgamma: float,
    transparency: bool,
    legend_size: float,
    legend_scale: float,
    legend_title: str,
    legend_title_color: Sequence[float],
    legend_position: Union[str, Sequence[int]],
    legend_alpha: float,
    devices: Sequence[int],
) -> None:
    """Renders datatset using 2D projections."""

    if "," in legend_position:
        legend_position = tuple(float(strvalue) for strvalue in legend_position.split(","))

    with asection(
        f"Projection rendering of: {input_dataset.path} to {output_path} for channels: {channels}, slicing: {input_dataset.slicing} "
    ):
        dataset_projection_rendering(
            input_dataset=input_dataset,
            output_path=output_path,
            channels=channels,
            overwrite=overwrite,
            devices=devices,
            axis=axis,
            dir=dir,
            mode=mode,
            clim=clim,
            attenuation=attenuation,
            gamma=gamma,
            dlim=dlim,
            cmap=colormap,
            rgb_gamma=rgbgamma,
            transparency=transparency,
            legend_size=legend_size,
            legend_scale=legend_scale,
            legend_title=legend_title,
            legend_title_color=legend_title_color,
            legend_position=legend_position,
            legend_alpha=legend_alpha,
        )

        input_dataset.close()
        aprint("Done!")
