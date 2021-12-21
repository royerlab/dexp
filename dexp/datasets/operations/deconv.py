from typing import Sequence, Tuple, List, Optional

import dask
import numpy
import scipy
import functools
from pathlib import Path

from arbol.arbol import aprint
from arbol.arbol import asection
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from dexp.datasets.base_dataset import BaseDataset
from dexp.optics.psf.standard_psfs import nikon16x08na, olympus20x10na
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.deconvolution.admm_deconvolution import admm_deconvolution
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.slicing import slice_from_shape


def dataset_deconv(dataset: BaseDataset,
                   dest_path: str,
                   channels: Sequence[str],
                   slicing,
                   store: str = 'dir',
                   compression: str = 'zstd',
                   compression_level: int = 3,
                   overwrite: bool = False,
                   tilesize: Optional[Tuple[int]] = None,
                   method: str = 'lr',
                   num_iterations: int = 16,
                   max_correction: int = 16,
                   power: float = 1,
                   blind_spot: int = 0,
                   back_projection: Optional[str] = None,
                   wb_order: int = 5,
                   psf_objective: str = 'nikon16x08na',
                   psf_na: float = 0.8,
                   psf_dxy: float = 0.485,
                   psf_dz: float = 2,
                   psf_xy_size: int = 17,
                   psf_z_size: int = 17,
                   psf_show: bool = False,
                   scaling: Optional[Tuple[float]] = None,
                   workers: int = 1,
                   workersbackend: str = '',
                   devices: Optional[List[int]] = None,
                   check: bool = True,
                   stop_at_exception: bool = True,
                   ):

    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(dest_path, mode, store, parent=dataset)

    # Default tile size:
    if tilesize is None:
        tilesize = 320 # very conservative

    # Scaling default value:
    if scaling is None:
        scaling = (1, 1, 1)
    sz, sy, sx = scaling
    aprint(f"Input images will be scaled by: (sz,sy,sx)={scaling}")

    # CUDA DASK cluster
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=devices)
    client = Client(cluster)
    aprint('Dask Client', client)

    lazy_computation = []

    for channel in dataset._selected_channels(channels):
        array = dataset.get_array(channel)

        aprint(f"Slicing with: {slicing}")
        out_shape, volume_slicing, time_points = slice_from_shape(array.shape, slicing)

        out_shape = tuple(int(round(u*v)) for u, v in zip(out_shape, (1,)+scaling))
        dtype = numpy.float16 if method == 'admm' else array.dtype 

        # Adds destination array channel to dataset
        dest_array = dest_dataset.add_channel(name=channel,
                                              shape=out_shape,
                                              dtype=dtype,
                                              codec=compression,
                                              clevel=compression_level)

        #This is not ideal but difficult to avoid right now:
        sxy = (sx+sy)/2

        # PSF paraneters:
        psf_kwargs = {'dxy': psf_dxy / sxy,
                      'dz': psf_dz / sz,
                      'xy_size': int(round(psf_xy_size * sxy)),
                      'z_size': int(round(psf_z_size * sz))}

        aprint(f"psf_kwargs: {psf_kwargs}")

        # NA override:
        if psf_na is not None:
            aprint(f"Numerical aperture overridden to a value of: {psf_na}")
            psf_kwargs['NA'] = psf_na

        # choose psf from detection optics:
        if psf_objective == 'nikon16x08na':
            psf_kernel = nikon16x08na(**psf_kwargs)
        elif psf_objective == 'olympus20x10na':
            psf_kernel = olympus20x10na(**psf_kwargs)
        elif Path(psf_objective).exists():
            psf_kernel = numpy.load(psf_objective)
            if sz != 1.0 or sy != 1.0 or sx != 1.0:
                psf_kernel = scipy.ndimage.interpolation.zoom(psf_kernel, zoom=(sz, sy, sx), order=1)
            psf_z_size = psf_kernel.shape[0] + 10
            psf_xy_size = max(psf_kernel.shape[1:]) + 10
        else:
            raise RuntimeError(f'Object/path {psf_objective} not found.')

        # usefull for debugging:
        if psf_show:
            import napari
            viewer = napari.Viewer(title=f"DEXP | viewing PSF with napari", ndisplay=3)
            viewer.add_image(psf_kernel)
            napari.run()

        margins = max(psf_xy_size, psf_z_size)

        if method == 'lr':
            normalize = False
            convolve = functools.partial(fft_convolve,
                                         in_place=False,
                                         mode='reflect',
                                         internal_dtype=numpy.float32)

            def deconv(image):
                min_value = image.min()
                max_value = image.max()

                return lucy_richardson_deconvolution(image=image,
                                                     psf=psf_kernel,
                                                     num_iterations=num_iterations,
                                                     max_correction=max_correction,
                                                     normalise_minmax=(min_value, max_value),
                                                     power=power,
                                                     blind_spot=blind_spot,
                                                     blind_spot_mode='median+uniform',
                                                     blind_spot_axis_exclusion=(0,),
                                                     wb_order=wb_order,
                                                     back_projection=back_projection,
                                                     convolve_method=convolve
                                                     )
        elif method == 'admm':
            normalize = True
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


        @dask.delayed
        def process(i):
            tp = time_points[i]
            try:
                with asection(f'Deconvolving time point for time point {i}/{len(time_points)}'):
                    with asection(f"Loading channel: {channel}"):
                        tp_array = numpy.asarray(array[tp][volume_slicing])

                    with BestBackend(exclusive=True, enable_unified_memory=True):

                        if sz != 1.0 or sy != 1.0 or sx != 1.0:
                            with asection(f"Applying scaling {(sz, sy, sx)} to image."):
                                sp = Backend.get_sp_module()
                                tp_array = Backend.to_backend(tp_array)
                                tp_array = sp.ndimage.interpolation.zoom(tp_array, zoom=(sz, sy, sx), order=1)
                                tp_array = Backend.to_numpy(tp_array)

                        with asection(f"Deconvolving image of shape: {tp_array.shape}, with tile size: {tilesize}, margins: {margins} "):
                            aprint(f"Number of iterations: {num_iterations}, back_projection:{back_projection}, ")
                            tp_array = scatter_gather_i2i(deconv, tp_array, tiles=tilesize, margins=margins,
                                                          normalise=normalize, internal_dtype=dtype)

                        with asection(f"Moving array from backend to numpy."):
                            tp_array = Backend.to_numpy(tp_array, dtype=dest_array.dtype, force_copy=False)

                    with asection(f"Saving deconvolved stack for time point {i}, shape:{tp_array.shape}, dtype:{array.dtype}"):
                        dest_dataset.write_stack(channel=channel,
                                                 time_point=i,
                                                 stack_array=tp_array)

                    aprint(f"Done processing time point: {i}/{len(time_points)} .")

            except Exception as error:
                aprint(error)
                aprint(f"Error occurred while processing time point {i} !")
                import traceback
                traceback.print_exc()

                if stop_at_exception:
                    raise error

        for i in range(len(time_points)):
            lazy_computation.append(process(i))

    dask.compute(*lazy_computation)

    # Dataset info:
    aprint(dest_dataset.info())

    # Check dataset integrity:
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
    client.close()
