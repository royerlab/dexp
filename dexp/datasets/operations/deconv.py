import functools
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import dask
import numpy as np
import scipy
from arbol.arbol import aprint, asection
from toolz import curry

from dexp.datasets import BaseDataset
from dexp.datasets.stack_iterator import StackIterator
from dexp.datasets.zarr_dataset import ZDataset
from dexp.optics.psf.standard_psfs import nikon16x08na, olympus20x10na
from dexp.processing.deconvolution import (
    admm_deconvolution,
    lucy_richardson_deconvolution,
)
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.backends import BestBackend
from dexp.utils.dask import get_dask_client


def get_psf(
    psf_objective: str,
    psf_na: float,
    psf_dxy: float,
    psf_dz: float,
    psf_z_size: int,
    psf_xy_size: int,
    scaling: Tuple[float],
    psf_show: bool,
) -> np.ndarray:

    # This is not ideal but difficult to avoid right now:
    sz, sy, sx = scaling

    if Path(psf_objective).exists():
        # loading pre-computed PSF
        psf_kernel = np.load(psf_objective)
        if sz != 1.0 or sy != 1.0 or sx != 1.0:
            psf_kernel = scipy.ndimage.zoom(psf_kernel, zoom=(sz, sy, sx), order=1)
    else:
        sxy = (sx + sy) / 2

        # PSF paraneters:
        psf_kwargs = {
            "dxy": psf_dxy / sxy,
            "dz": psf_dz / sz,
            "xy_size": int(round(psf_xy_size * sxy)),
            "z_size": int(round(psf_z_size * sz)),
        }

        aprint(f"psf_kwargs: {psf_kwargs}")

        # NA override:
        if psf_na is not None:
            aprint(f"Numerical aperture overridden to a value of: {psf_na}")
            psf_kwargs["NA"] = psf_na

        # choose psf from detection optics:
        if psf_objective == "nikon16x08na":
            psf_kernel = nikon16x08na(**psf_kwargs)
        elif psf_objective == "olympus20x10na":
            psf_kernel = olympus20x10na(**psf_kwargs)
        else:
            raise RuntimeError(f"Object/path {psf_objective} not found.")

    # usefull for debugging:
    if psf_show:
        import napari

        viewer = napari.Viewer(title="DEXP | viewing PSF with napari", ndisplay=3)
        viewer.add_image(psf_kernel)
        napari.run()

    return psf_kernel


def get_deconv_func(
    method: str,
    psf_kernel: np.ndarray,
    num_iterations: int,
    max_correction: int,
    blind_spot: int,
    power: float,
    wb_order: int,
    back_projection: str,
) -> Callable:

    if method == "lr" or method == "wb":
        convolve = functools.partial(fft_convolve, in_place=False, mode="reflect", internal_dtype=np.float32)

        def deconv(image):
            min_value = image.min()
            max_value = image.max()

            return lucy_richardson_deconvolution(
                image=image,
                psf=psf_kernel,
                num_iterations=num_iterations,
                max_correction=max_correction,
                normalise_minmax=(min_value, max_value),
                power=power,
                blind_spot=blind_spot,
                blind_spot_mode="median+uniform",
                blind_spot_axis_exclusion=(0,),
                wb_order=wb_order,
                back_projection=back_projection,
                convolve_method=convolve,
            )

    elif method == "admm":

        def deconv(image):
            out = admm_deconvolution(
                image,
                psf=psf_kernel,
                iterations=num_iterations,
                derivative=2,
            )
            return out

    else:
        raise ValueError(f"Unknown deconvolution mode: {method}")

    return deconv


@dask.delayed
@curry
def _process(
    time_point: int,
    stacks: StackIterator,
    out_dataset: ZDataset,
    channel: str,
    deconv_func: Callable,
    scaling: Tuple[int],
):

    with asection(f"Deconvolving time point for time point {time_point}/{len(stacks)}"):
        with asection(f"Loading channel: {channel}"):
            stack = np.asarray(stacks[time_point])

        with BestBackend() as bkd:
            if any(s != 1.0 for s in scaling):
                with asection(f"Applying scaling {scaling} to image."):
                    sp = bkd.get_sp_module()
                    stack = bkd.to_backend(stack)
                    stack = sp.ndimage.zoom(stack, zoom=scaling, order=1)
                    stack = bkd.to_numpy(stack)

            with asection("Deconvolving ..."):
                stack = deconv_func(stack)

                with asection("Moving array from backend to numpy."):
                    stack = bkd.to_numpy(stack, dtype=out_dataset.dtype(channel), force_copy=False)

            with asection(
                f"Saving deconvolved stack for time point {time_point}, shape:{stack.shape}, dtype:{stack.dtype}"
            ):
                out_dataset.write_stack(channel=channel, time_point=time_point, stack_array=stack)

            aprint(f"Done processing time point: {time_point}/{len(stacks)} .")


def dataset_deconv(
    input_dataset: BaseDataset,
    output_dataset: ZDataset,
    channels: Sequence[str],
    tilesize: int,
    method: str,
    num_iterations: Optional[int],
    max_correction: int,
    power: float,
    blind_spot: int,
    back_projection: Optional[str],
    wb_order: int,
    psf_objective: str,
    psf_na: Optional[float],
    psf_dxy: float,
    psf_dz: float,
    psf_xy_size: int,
    psf_z_size: int,
    psf_show: bool,
    scaling: Tuple[float],
    devices: List[int],
):
    aprint(f"Input images will be scaled by: (sz,sy,sx)={scaling}")

    psf_kernel = get_psf(psf_objective, psf_na, psf_dxy, psf_dz, psf_z_size, psf_xy_size, scaling, psf_show)
    margins = max(psf_kernel.shape[1], psf_kernel.shape[0])

    deconv_func = get_deconv_func(
        method,
        psf_kernel,
        num_iterations,
        max_correction,
        blind_spot,
        power,
        wb_order,
        back_projection,
    )
    deconv_func = curry(
        scatter_gather_i2i,
        function=deconv_func,
        tiles=tilesize,
        margins=margins,
        normalise=method == "admm",
    )

    lazy_computation = []

    for channel in channels:
        stacks = input_dataset[channel]

        out_shape = tuple(int(round(u * v)) for u, v in zip(stacks.shape, (1,) + scaling))
        dtype = np.float16 if method == "admm" else stacks.dtype

        # Adds destination array channel to dataset
        output_dataset.add_channel(name=channel, shape=out_shape, dtype=dtype)

        process = _process(
            stacks=stacks,
            out_dataset=output_dataset,
            channel=channel,
            scaling=scaling,
            deconv_func=deconv_func(internal_dtype=dtype),
        )

        for t in range(len(stacks)):
            lazy_computation.append(process(time_point=t))

    dask.compute(*lazy_computation)

    # CUDA DASK cluster
    client = get_dask_client(devices)
    aprint("Dask Client", client)

    # Dataset info:
    aprint(output_dataset.info())

    # Check dataset integrity:
    output_dataset.check_integrity()
