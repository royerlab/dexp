from typing import List, Optional, Sequence, Tuple

import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import (
    channels_option,
    input_dataset_argument,
    multi_devices_option,
    output_dataset_options,
    slicing_option,
    tilesize_option,
    tuple_callback,
)
from dexp.datasets.base_dataset import BaseDataset
from dexp.datasets.operations.deconv import dataset_deconv
from dexp.datasets.zarr_dataset import ZDataset


@click.command()
@input_dataset_argument()
@output_dataset_options()
@channels_option()
@multi_devices_option()
@slicing_option()
@tilesize_option()
@click.option(
    "--method",
    "-m",
    type=str,
    default="lr",
    help="Deconvolution method: for now only lr (Lucy Richardson)",
    show_default=True,
)
@click.option(
    "--iterations",
    "-i",
    type=int,
    default=5,
    help="Number of deconvolution iterations. More iterations takes longer, will be sharper, but might also be potentially more noisy depending on method. "
    "The default number of iterations depends on the other parameters, in particular it depends on the choice of backprojection operator. For ‘wb’ as little as 3 iterations suffice. ",
    show_default=True,
)
@click.option(
    "--max-correction",
    "-mc",
    type=int,
    default=None,
    help="Max correction in folds per iteration. By default there is no limit",
    show_default=True,
)
@click.option(
    "--power",
    "-pw",
    type=float,
    default=1.0,
    help="Correction exponent, default for standard LR is 1, set to >1 for acceleration.",
    show_default=True,
)
@click.option(
    "--blind-spot",
    "-bs",
    type=int,
    default=0,
    help="Blindspot based noise reduction. Provide size of kernel to use, must be an odd number: 3(recommended), 5, 7. 0 means no blindspot. ",
    show_default=True,
)
@click.option(
    "--back-projection",
    "-bp",
    type=click.Choice(["tpsf", "wb"]),
    default="tpsf",
    help="Back projection operator, can be: ‘tpsf’ (transposed PSF = classic) or ‘wb’ (Wiener-Butterworth =  accelerated) ",
    show_default=True,
)
@click.option(
    "--wb-order",
    "-wbo",
    type=int,
    default=5,
    help="Wiener-Butterworth order parameter, a higher `n` makes the filter transition slope closer to a hard cutoff, causing the ringing artifacts in the spatial domain."
    "In contrast, a lower order `n` makes the transition slope gentler and sacrifice some spectral amplitude at spatial frequencies close to the cutoff.",
    show_default=True,
)
@click.option(
    "--objective",
    "-obj",
    type=str,
    default="nikon16x08na",
    help="Microscope objective to use for computing psf, can be: nikon16x08na, olympus20x10na, or path to file",
    show_default=True,
)
@click.option("--num-ape", "-na", type=float, default=None, help="Overrides NA value for objective.", show_default=True)
@click.option("--dxy", "-dxy", type=float, default=0.485, help="Voxel size along x and y in microns", show_default=True)
@click.option("--dz", "-dz", type=float, default=4 * 0.485, help="Voxel size along z in microns", show_default=True)
@click.option("--xy-size", "-sxy", type=int, default=31, help="PSF size along xy in voxels", show_default=True)
@click.option("--z-size", "-sz", type=int, default=31, help="PSF size along z in voxels", show_default=True)
@click.option("--show-psf", "-sp", is_flag=True, help="Show point spread function (PSF) with napari", show_default=True)
@click.option(
    "--scaling",
    "-sc",
    type=str,
    default="1,1,1",
    help="Scales input image along the three axis: sz,sy,sx (numpy order). For example: 2,1,1 upscales along z by a factor 2",
    show_default=True,
    callback=tuple_callback(dtype=float, length=3),
)
def deconv(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    tilesize: int,
    method: str,
    iterations: Optional[int],
    max_correction: int,
    power: float,
    blind_spot: int,
    back_projection: Optional[str],
    wb_order: int,
    objective: str,
    num_ape: Optional[float],
    dxy: float,
    dz: float,
    xy_size: int,
    z_size: int,
    show_psf: bool,
    scaling: Tuple[float],
    devices: List[int],
):
    """Deconvolves all or selected channels of a dataset."""

    with asection(
        f"Deconvolving dataset: {input_dataset.path}, saving it at: {output_dataset.path}, for channels: {channels}, slicing: {input_dataset.slicing} "
    ):
        dataset_deconv(
            input_dataset,
            output_dataset,
            channels=channels,
            tilesize=tilesize,
            method=method,
            num_iterations=iterations,
            max_correction=max_correction,
            power=power,
            blind_spot=blind_spot,
            back_projection=back_projection,
            wb_order=wb_order,
            psf_objective=objective,
            psf_na=num_ape,
            psf_dxy=dxy,
            psf_dz=dz,
            psf_xy_size=xy_size,
            psf_z_size=z_size,
            psf_show=show_psf,
            scaling=scaling,
            devices=devices,
        )

        input_dataset.close()
        output_dataset.close()
        aprint("Done!")
