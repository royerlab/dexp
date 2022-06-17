from typing import Sequence

import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import (
    channels_option,
    input_dataset_argument,
    multi_devices_option,
    slicing_option,
    verbose_option,
)
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.operations.extract_psf import dataset_extract_psf


@click.command(name="extract-psf")
@input_dataset_argument()
@slicing_option()
@channels_option()
@multi_devices_option()
@verbose_option()
@click.option("--out-prefix-path", "-o", default="psf_", help="Output PSF file prefix", type=str)
@click.option(
    "--peak-threshold",
    "-pt",
    type=int,
    default=500,
    show_default=True,
    help="Peak valeu threshold for object (PSF) detection. Lower values are less consertaive and will detect more objects.",
)
@click.option(
    "--similarity-threshold",
    "-st",
    type=float,
    default=0.5,
    show_default=True,
    help="Threshold of PSF selection given the similarity (cosine distance) to median PSF.",
)
@click.option("--psf_size", "-ps", type=int, default=35, show_default=True, help="Size (shape) of the PSF")
def extract_psf(
    input_dataset: BaseDataset,
    out_prefix_path: str,
    channels: Sequence[str],
    peak_threshold: int,
    similarity_threshold: float,
    psf_size: int,
    devices: Sequence[int],
    verbose: bool,
):
    """Detects and extracts the PSF from beads."""
    with asection(
        f"Extracting PSF of dataset: {input_dataset.path}, saving it with prefix: {out_prefix_path}, for channels: {channels}"
    ):
        aprint(f"Device used: {devices}")
        dataset_extract_psf(
            input_dataset=input_dataset,
            dest_path=out_prefix_path,
            channels=channels,
            peak_threshold=peak_threshold,
            similarity_threshold=similarity_threshold,
            psf_size=psf_size,
            verbose=verbose,
            devices=devices,
        )
