from typing import Optional, Sequence

import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import DEFAULT_WORKERS_BACKEND
from dexp.cli.parsing import (
    channels_option,
    device_option,
    input_dataset_argument,
    output_dataset_options,
    workers_option,
)
from dexp.datasets.operations.stabilize import dataset_stabilize
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@input_dataset_argument()
@output_dataset_options()
@channels_option()
@click.option(
    "--reference-channel", "-rc", default=None, help="Reference channel for single stabilization model computation."
)
@workers_option()
@device_option()
@click.option(
    "--maxrange",
    "-mr",
    type=click.IntRange(min=1),
    default=7,
    help="Maximal distance, in time points, between pairs of images to registrate.",
    show_default=True,
)
@click.option(
    "--minconfidence",
    "-mc",
    type=click.FloatRange(min=0, max=1),
    default=0.5,
    help="Minimal confidence for registration parameters, if below that level the registration parameters for previous time points is used.",
    show_default=True,
)
@click.option(
    "--com/--no-com",
    type=bool,
    default=False,
    is_flag=True,
    help="Enable center of mass fallback when standard registration fails.",
    show_default=True,
)
@click.option(
    "--quantile",
    "-q",
    type=click.FloatRange(min=0, max=1),
    default=0.5,
    help="Quantile to cut-off background in center-of-mass calculation.",
    show_default=True,
)
@click.option("--tolerance", "-t", type=float, default=1e-7, help="Tolerance for linear solver.", show_default=True)
@click.option(
    "--ordererror", "-oe", type=float, default=2.0, help="Order for linear solver error term.", show_default=True
)
@click.option(
    "--orderreg", "-or", type=float, default=1.0, help="Order for linear solver regularisation term.", show_default=True
)
@click.option(
    "--alphareg",
    "-or",
    type=float,
    default=1e-4,
    help="Multiplicative coefficient for regularisation term.",
    show_default=True,
)
@click.option(
    "--pcsigma",
    "-rs",
    type=float,
    default=2,
    help="Sigma for Gaussian smoothing of phase correlogram, zero to disable.",
    show_default=True,
)
@click.option(
    "--dsigma",
    "-ds",
    type=float,
    default=1.5,
    help="Sigma for Gaussian smoothing (crude denoising) of input images, zero to disable.",
    show_default=True,
)
@click.option(
    "--logcomp",
    "-lc",
    type=bool,
    default=True,
    help="Applies the function log1p to the images to compress high-intensities (usefull when very (too) bright structures are present in the images, such as beads.",
    show_default=True,
)
@click.option(
    "--edgefilter",
    "-ef",
    type=bool,
    default=False,
    is_flag=True,
    help="Applies sobel edge filter to input images.",
    show_default=True,
)
@click.option(
    "--detrend",
    "-dt",
    type=bool,
    is_flag=True,
    default=False,
    help="Remove linear trend from stabilization result",
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
    "--model-input-path",
    "-mi",
    type=str,
    default=None,
    help="Path to pre-computed model for image registration",
)
@click.option(
    "--model-output-path",
    "-mo",
    type=str,
    default="stabilization_model.json",
    show_default=True,
    help="Output path for computed registration model",
)
@click.option(
    "--workersbackend",
    "-wkb",
    type=str,
    default=DEFAULT_WORKERS_BACKEND,
    help="What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ",
    show_default=True,
)
@click.option("--device", "-d", type=int, default=0, help="Sets the CUDA devices id, e.g. 0,1,2", show_default=True)  #
def stabilize(
    input_dataset: ZDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    reference_channel: Optional[str],
    maxrange: int,
    minconfidence: float,
    com: bool,
    quantile: float,
    tolerance: float,
    ordererror: float,
    orderreg: float,
    alphareg: float,
    pcsigma: float,
    dsigma: float,
    logcomp: bool,
    edgefilter: bool,
    detrend: bool,
    maxproj: bool,
    model_input_path: str,
    model_output_path: str,
    workers: int,
    workersbackend: str,
    device: int,
):
    """Stabilises dataset against translations across time."""
    with asection(
        f"Stabilizing dataset(s): {input_dataset.path}, saving it at: {output_dataset.path}, for channels: {channels}"
    ):
        dataset_stabilize(
            input_dataset,
            output_dataset,
            channels=channels,
            model_output_path=model_output_path,
            model_input_path=model_input_path,
            reference_channel=reference_channel,
            max_range=maxrange,
            min_confidence=minconfidence,
            enable_com=com,
            quantile=quantile,
            tolerance=tolerance,
            order_error=ordererror,
            order_reg=orderreg,
            alpha_reg=alphareg,
            phase_correlogram_sigma=pcsigma,
            denoise_input_sigma=dsigma,
            log_compression=logcomp,
            edge_filter=edgefilter,
            detrend=detrend,
            maxproj=maxproj,
            workers=workers,
            workers_backend=workersbackend,
            device=device,
        )

        input_dataset.close()
        output_dataset.close()
        aprint("Done!")
