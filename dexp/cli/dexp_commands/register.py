from typing import Sequence

import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import (
    channels_option,
    input_dataset_argument,
    multi_devices_option,
    slicing_option,
)
from dexp.datasets import BaseDataset
from dexp.datasets.operations.register import dataset_register


@click.command()
@input_dataset_argument()
@slicing_option()
@channels_option()
@multi_devices_option()
@click.option("--out-model-path", "-o", default="registration_models.txt", show_default=True)
@click.option(
    "--microscope",
    "-m",
    type=click.Choice(("simview", "mvsols")),
    default="simview",
    help="Microscope used to acquire the data, this selects the fusion algorithm.",
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
@click.option(
    "--fusion", "-f", type=click.Choice(("tg", "dct", "dft")), default="tg", help="Fusion method.", show_default=True
)
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
    input_dataset: BaseDataset,
    out_model_path: str,
    channels: Sequence[str],
    microscope: str,
    equalise: bool,
    zero_level: int,
    clip_high: int,
    fusion: str,
    fusion_bias_strength: float,
    dehaze_size: int,
    edge_filter: bool,
    max_proj: bool,
    white_top_hat_size: float,
    white_top_hat_sampling: int,
    remove_beads: bool,
    devices: Sequence[int],
) -> None:
    """
    Computes registration model for fusing.
    """
    with asection(
        f"Computing fusion model of dataset: {input_dataset.path}, saving it at: {out_model_path}, for channels: {channels}"
    ):
        aprint(f"Microscope type: {microscope}, fusion type: {fusion}")
        aprint(f"Devices used: {devices}")
        dataset_register(
            dataset=input_dataset,
            model_path=out_model_path,
            channels=channels,
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

    input_dataset.close()
