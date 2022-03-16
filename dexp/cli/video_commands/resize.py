import click

from dexp.cli.defaults import DEFAULT_WORKERS_BACKEND
from dexp.video.crop_resize_pad import crop_resize_pad_image_sequence


@click.command()
@click.argument("input_path", type=str)
@click.option("--output_path", "-o", type=str, default=None, help="Output folder for resized frames.")
@click.option(
    "--crop",
    "-c",
    type=str,
    default=None,
    help="Crop image by removing a given number of pixels/voxels per axis."
    " For example: 10,20*10,20 crops 10 pixels on the left for axis 0, 20 pixels"
    " from the right of axis #0, and the same for axis 2.",
    show_default=True,
)
@click.option(
    "--resize",
    "-r",
    type=str,
    default=None,
    help="After cropping, the image is resized to the given shape."
    " If any entry in the tuple is -1 then that position in the shape"
    " is automatically determined based on the existing shape to preserve aspect ratio.",
    show_default=True,
)
@click.option(
    "--resizeorder",
    "-ro",
    type=int,
    default=3,
    help="The order of the spline interpolation, default is 3. The order has to be in the range 0-5.",
    show_default=True,
)
@click.option(
    "--resizemode",
    "-rm",
    type=str,
    default="constant",
    help="Optional The mode parameter determines how the input array"
    " is extended beyond its boundaries. Can be: ‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’.",
    show_default=True,
)
@click.option(
    "--padwidth",
    "-pw",
    type=str,
    default=None,
    help="After cropping and resizing, padding is performed."
    " The provided tuple is interpreted similarly to cropping."
    " For example, 5,5*5,5 pads all sides with 5 pixels.",
    show_default=True,
)
@click.option(
    "--padmode",
    "-pm",
    type=str,
    default="constant",
    help="Padding mode. See numpy.pad for the available modes",
    show_default=True,
)
@click.option(
    "--padcolor",
    "-pc",
    type=str,
    default="0,0,0,0",
    help="Padding color as tuple of normalised floats:  (R,G,B,A). Default is transparent black.",
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
def resize(
    input_path,
    output_path,
    crop,
    resize,
    resizeorder,
    resizemode,
    padwidth,
    padmode,
    padcolor,
    overwrite,
    workers,
    workersbackend,
    device,
):
    """Resizes video"""

    # Default output path:
    if output_path is None:
        output_path = input_path + "_resized"
    elif output_path.startswith("_"):
        output_path = input_path + output_path

    if crop is not None:
        crop = tuple(tuple(int(v) for v in xy.split(",")) for xy in crop.split("*"))

    if resize is not None and "," in resize:
        resize = tuple(int(v) for v in resize.split(","))

    if padwidth is not None:
        padwidth = tuple(tuple(int(v) for v in xy.split(",")) for xy in padwidth.split("*"))

    # Parse color:
    if padcolor is not None and "," in padcolor:
        padcolor = tuple(float(v) for v in padcolor.split(","))

    crop_resize_pad_image_sequence(
        input_path=input_path,
        output_path=output_path,
        crop=crop,
        resize=resize,
        resize_order=resizeorder,
        resize_mode=resizemode,
        pad_width=padwidth,
        pad_mode=padmode,
        pad_color=padcolor,
        overwrite=overwrite,
        workers=workers,
        workersbackend=workersbackend,
        device=device,
    )
