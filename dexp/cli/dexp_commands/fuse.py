from typing import Sequence, Tuple

import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import (
    channels_option,
    input_dataset_argument,
    multi_devices_option,
    output_dataset_options,
    slicing_option,
    tuple_callback,
)
from dexp.datasets import BaseDataset, ZDataset
from dexp.datasets.operations.fuse import dataset_fuse


@click.command()
@input_dataset_argument()
@output_dataset_options()
@channels_option()
@slicing_option()
@multi_devices_option()
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
    "--equalisemode",
    "-eqm",
    default="first",
    type=click.Choice(["first", "all"]),
    help="Equalisation modes: compute correction ratios only for first time point: ‘first’ or for all time points: ‘all’.",
    show_default=True,
)
@click.option(
    "--zerolevel",
    "-zl",
    type=int,
    default=0,
    help="‘zero-level’ i.e. the pixel values in the restoration (to be substracted)",
    show_default=True,
)
@click.option(
    "--cliphigh",
    "-ch",
    type=int,
    default=0,
    help="Clips voxel values above the given value, if zero no clipping is done",
    show_default=True,
)
@click.option(
    "--fusion",
    "-f",
    type=click.Choice(["tg", "dct"]),
    default="tg",
    help="Fusion mode, can be: ‘tg’ or ‘dct’.",
    show_default=True,
)
@click.option(
    "--fusion_bias_strength",
    "-fbs",
    type=str,
    default="0.5,0.02",
    help="Fusion bias strength for illumination and detection ‘fbs_i fbs_d’, set to ‘0 0’) if fusing a cropped region",
    show_default=True,
    callback=tuple_callback(dtype=float, length=2),
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
    "--dark-denoise-threshold",
    "-ddt",
    type=int,
    default=0,
    help="Threshold for denoises the dark pixels of the image -- helps increase compression ratio. Set to zero to deactivate.",
    show_default=True,
)
@click.option(
    "--zpadapodise",
    "-zpa",
    type=str,
    default="8,96",
    help="Pads and apodises the views along z before fusion: ‘pad apo’, where pad is a padding length, and apo is apodisation length, both in voxels. If pad=apo, no original voxel is modified and only added voxels are apodised.",
    show_default=True,
    callback=tuple_callback(length=2, dtype=int),
)
@click.option(
    "--loadreg",
    "-lr",
    is_flag=True,
    help="Turn on to load the registration parameters from a previous run",
    show_default=True,
)
@click.option(
    "--model-filename",
    "-mf",
    help="Model filename to load or save registration model list",
    default="registration_models.txt",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    "--warpregiter",
    "-wri",
    type=int,
    default=4,
    help="Number of iterations for warp registration (if applicable).",
    show_default=True,
)
@click.option(
    "--minconfidence",
    "-mc",
    type=float,
    default=0.3,
    help="Minimal confidence for registration parameters, if below that level the registration parameters for previous time points is used.",
    show_default=True,
)
@click.option(
    "--maxchange",
    "-md",
    type=float,
    default=16,
    help="Maximal change in registration parameters, if above that level the registration parameters for previous time points is used.",
    show_default=True,
)
@click.option(
    "--regedgefilter",
    "-ref",
    is_flag=True,
    help="Use this flag to apply an edge filter to help registration.",
    show_default=True,
)
@click.option(
    "--maxproj/--no-maxproj",
    "-mp/-nmp",
    type=bool,
    default=True,
    help="Registers using only the maximum intensity projection from each stack.",
    show_default=True,
)
@click.option(
    "--hugedataset",
    "-hd",
    is_flag=True,
    help="Use this flag to indicate that the the dataset is _huge_ and that memory allocation should be optimised at the detriment of processing speed.",
    show_default=True,
)
@click.option(
    "--pad", "-p", is_flag=True, default=False, help="Use this flag to pad views according to the registration models."
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
def fuse(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    microscope: str,
    equalise: bool,
    equalisemode: str,
    zerolevel: float,
    cliphigh: float,
    fusion: str,
    fusion_bias_strength: Tuple[float, float],
    dehaze_size: int,
    dark_denoise_threshold: int,
    zpadapodise: Tuple[float, float],
    loadreg: bool,
    model_filename: str,
    warpregiter: int,
    minconfidence: float,
    maxchange: float,
    regedgefilter: bool,
    maxproj: bool,
    hugedataset: bool,
    devices: Sequence[int],
    pad: bool,
    white_top_hat_size: float,
    white_top_hat_sampling: int,
    remove_beads: bool,
):
    """Fuses the views of a multi-view light-sheet microscope dataset (available: simview and mvsols)"""

    with asection(
        f"Fusing dataset: {input_dataset.path}, saving it at: {output_dataset.path},"
        + f"for channels: {channels}, slicing: {input_dataset.slicing}"
    ):
        aprint(f"Microscope type: {microscope}, fusion type: {fusion}")
        aprint(f"Devices used: {devices}")
        dataset_fuse(
            input_dataset=input_dataset,
            output_dataset=output_dataset,
            channels=channels,
            microscope=microscope,
            equalise=equalise,
            equalise_mode=equalisemode,
            zero_level=zerolevel,
            clip_too_high=cliphigh,
            fusion=fusion,
            fusion_bias_strength_i=fusion_bias_strength[0],
            fusion_bias_strength_d=fusion_bias_strength[1],
            dehaze_size=dehaze_size,
            dark_denoise_threshold=dark_denoise_threshold,
            z_pad_apodise=zpadapodise,
            loadreg=loadreg,
            model_list_filename=model_filename,
            warpreg_num_iterations=warpregiter,
            min_confidence=minconfidence,
            max_change=maxchange,
            registration_edge_filter=regedgefilter,
            maxproj=maxproj,
            huge_dataset=hugedataset,
            devices=devices,
            pad=pad,
            white_top_hat_size=white_top_hat_size,
            white_top_hat_sampling=white_top_hat_sampling,
            remove_beads=remove_beads,
        )

        input_dataset.close()
        output_dataset.close()
        aprint("Done!")
