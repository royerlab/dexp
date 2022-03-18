import click

from dexp.cli.defaults import DEFAULT_WORKERS_BACKEND
from dexp.video.overlay import add_overlays_image_sequence


@click.command()
@click.argument("input_path", type=str)
@click.option("--output_path", "-o", type=str, default=None, help="Output folder for overlayed frames.")
@click.option("--scalebar/--no-scalebar", "-sb/-nsb", default=True, help="True to insert scale bar.", show_default=True)
@click.option(
    "--barlength", "-bl", type=float, default=1, help="Length of scale bar in the provided unit.", show_default=True
)
@click.option(
    "--barscale",
    "-bs",
    type=float,
    default=1,
    help="Conversion factor from pixels to units -- what is the side length of a pixel/voxel in units.",
    show_default=True,
)
@click.option("--barheight", "-bh", type=int, default=4, help="Height of th scale bar in pixels", show_default=True)
@click.option(
    "--barpos",
    "-bp",
    type=str,
    default="bottom_right",
    help="Positions of the scale bar in pixels in natural order: (x, y)."
    " Can also be a string: bottom_left, bottom_right, top_left, top_right.",
    show_default=True,
)
@click.option("--barunit", "-bu", type=str, default="μm", help="Scale bar unit name.", show_default=True)
@click.option(
    "--timestamp/--no-timestamp", "-ts/-nts", default=True, help="True to insert time stamp.", show_default=True
)
@click.option("--timestart", "-ts", type=float, default=0, help="Start time for time stamp", show_default=True)
@click.option(
    "--timeinterval",
    "-ti",
    type=float,
    default=1,
    help="Time interval inn units of time between consecutive images.",
    show_default=True,
)
@click.option(
    "--timepos",
    "-tp",
    type=str,
    default="top_right",
    help="Positions of the time stamp in pixels in natural order: (x, y)."
    " Can also be a string: bottom_left, bottom_right, top_left, top_right.",
    show_default=True,
)
@click.option("--timeunit", "-tu", type=str, default="s", help="Time stamp time unit name.", show_default=True)
@click.option(
    "--margin",
    "-mg",
    type=float,
    default=1,
    help="Margin around bar expressed in units relative to the text height",
    show_default=True,
)
@click.option(
    "--color",
    "-c",
    type=str,
    default=None,
    help="Color of the bar and text as tuple of 4 values: (R, G, B, A)",
    show_default=True,
)
@click.option(
    "--numberformat",
    "-nf",
    type=str,
    default="{:.1f}",
    help="Format string to represent the start and end values.",
    show_default=True,
)
@click.option("--fontname", "-fn", type=str, default="Helvetica", help="Font name.", show_default=True)
@click.option("--fontsize", "-fs", type=int, default=32, help="Font size in pixels.", show_default=True)
@click.option(
    "--mode",
    "-md",
    type=str,
    default="max",
    help="Blending modes. Either one for all images, or one per image in the form of a sequence."
    " Blending modes are: mean, add, satadd, max, alpha.",
    show_default=True,
)
@click.option("--overwrite", "-w", is_flag=True, help="Force overwrite of output images.", show_default=True)
@click.option(
    "--workers",
    "-k",
    type=int,
    default=-1,
    help="Number of worker threads to spawn, set to -1 for maximum number of workers",
    show_default=True,
)  #
@click.option(
    "--workersbackend",
    "-wkb",
    type=str,
    default=DEFAULT_WORKERS_BACKEND,
    help="What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ",
    show_default=True,
)  #
@click.option("--device", "-d", type=int, default=0, help="Sets the CUDA devices id, e.g. 0,1,2", show_default=True)  #
def overlay(
    input_path: str,
    output_path: str,
    scalebar,
    barlength,
    barscale,
    barheight,
    barpos,
    barunit,
    timestamp,
    timestart,
    timeinterval,
    timepos,
    timeunit,
    margin,
    color,
    numberformat,
    fontname,
    fontsize,
    mode,
    overwrite,
    workers,
    workersbackend,
    device,
):
    """Adds scale bars and timestamps to videos."""

    # Default output path:
    if output_path is None:
        output_path = input_path + "_overlay"
    elif output_path.startswith("_"):
        output_path = input_path + output_path

    # Parse bar position:
    if "," in barpos:
        barpos = tuple(float(v) for v in barpos.split(","))

    # Parse time position:
    if "," in timepos:
        timepos = tuple(float(v) for v in timepos.split(","))

    # Parse color:
    if color is not None and "," in color:
        color = tuple(float(v) for v in color.split(","))

    add_overlays_image_sequence(
        input_path=input_path,
        output_path=output_path,
        scale_bar=scalebar,
        scale_bar_length_in_unit=barlength,
        scale_bar_pixel_scale=barscale,
        scale_bar_bar_height=barheight,
        scale_bar_translation=barpos,
        scale_bar_unit=barunit,
        time_stamp=timestamp,
        time_stamp_start_time=timestart,
        time_stamp_time_interval=timeinterval,
        time_stamp_translation=timepos,
        time_stamp_unit=timeunit,
        margin=margin,
        color=color,
        number_format=numberformat,
        font_name=fontname,
        font_size=fontsize,
        mode=mode,
        overwrite=overwrite,
        workers=workers,
        workersbackend=workersbackend,
        device=device,
    )
