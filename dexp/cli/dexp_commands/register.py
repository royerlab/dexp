import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import _parse_channels, _parse_slicing, parse_devices
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.register import dataset_register


@click.command()
@click.argument("input_paths", nargs=-1, required=True)
@click.option("--out-model-path", "-o", default="registration_models.txt", show_default=True)
@click.option(
    "--channels",
    "-c",
    default=None,
    help="list of channels for the view in standard order for the microscope type (C0L0, C0L1, C1L0, C1L1,...)",
)
@click.option(
    "--slicing",
    "-s",
    default=None,
    help="dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ",
)
@click.option(
    "--microscope",
    "-m",
    type=str,
    default="simview",
    help="Microscope objective to use for computing psf, can be: simview or mvsols",
    show_default=True,
)
@click.option(
    "--equalise/--no-equalise",
    "-eq/-neq",
    default=True,
    help="Equalise intensity of views before fusion, or not.",
    show_default=True,
)
@click.option(
    "--zero-level",
    "-zl",
    type=int,
    default=0,
    help="‘zero-level’ i.e. the pixel values in the restoration (to be substracted)",
    show_default=True,
)
@click.option(
    "--clip-high",
    "-ch",
    type=int,
    default=0,
    help="Clips voxel values above the given value, if zero no clipping is done",
    show_default=True,
)
@click.option("--fusion", "-f", type=str, default="tg", help="Fusion mode, can be: ‘tg’ or ‘dct’.  ", show_default=True)
@click.option(
    "--fusion_bias_strength",
    "-fbs",
    type=float,
    default=0.5,
    help="Fusion bias strength for illumination",
    show_default=True,
)
@click.option(
    "--dehaze_size",
    "-dhs",
    type=int,
    default=65,
    help="Filter size (scale) for dehazing the final regsitered and fused image to reduce effect of scattered and out-of-focus light. Set to zero to deactivate.",
    show_default=True,
)
@click.option(
    "--edge-filter",
    "-ef",
    is_flag=True,
    help="Use this flag to apply an edge filter to help registration.",
    show_default=True,
)
@click.option(
    "--max-proj/--no-max-proj",
    "-mp/-nmp",
    type=bool,
    default=True,
    help="Registers using only the maximum intensity projection from each stack.",
    show_default=True,
)
@click.option(
    "--devices", "-d", type=str, default="0", help="Sets the CUDA devices id, e.g. 0,1,2 or ‘all’", show_default=True
)
@click.option(
    "--white-top-hat-size",
    "-wth",
    default=0,
    type=float,
    help="Area opening value after down sampling for white top hat transform transform. Larger values will keep larger components. Recommended value of 1e5.",
)
@click.option(
    "--white-top-hat-sampling", "-wths", default=4, type=int, help="Down sampling size to compute the area opening"
)
@click.option(
    "--remove-beads",
    "-rb",
    is_flag=True,
    default=False,
    help="Use this flag to remove beads before equalizing and fusing",
)
def register(
    input_paths,
    out_model_path,
    channels,
    slicing,
    microscope,
    equalise,
    zero_level,
    clip_high,
    fusion,
    fusion_bias_strength,
    dehaze_size,
    edge_filter,
    max_proj,
    white_top_hat_size,
    white_top_hat_sampling,
    remove_beads,
    devices,
):
    """
    Computes registration model for fusing.
    """
    input_dataset, input_paths = glob_datasets(input_paths)
    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)
    devices = parse_devices(devices)

    with asection(
        f"Fusing dataset: {input_paths}, saving it at: {out_model_path}, for channels: {channels}, slicing: {slicing} "
    ):
        aprint(f"Microscope type: {microscope}, fusion type: {fusion}")
        aprint(f"Devices used: {devices}")
        dataset_register(
            dataset=input_dataset,
            model_path=out_model_path,
            channels=channels,
            slicing=slicing,
            microscope=microscope,
            equalise=equalise,
            zero_level=zero_level,
            clip_too_high=clip_high,
            fusion=fusion,
            fusion_bias_strength_i=fusion_bias_strength,
            dehaze_size=dehaze_size,
            registration_edge_filter=edge_filter,
            white_top_hat_size=white_top_hat_size,
            white_top_hat_sampling=white_top_hat_sampling,
            remove_beads=remove_beads,
            max_proj=max_proj,
            devices=devices,
        )
