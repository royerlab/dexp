from typing import Sequence, Tuple, List

import numpy
from arbol.arbol import aprint
from arbol.arbol import asection

import dask
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from dexp.datasets.base_dataset import BaseDataset
from dexp.optics.psf.standard_psfs import nikon16x08na, olympus20x10na
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i


def dataset_deconv(dataset: BaseDataset,
                   path: str,
                   channels: Sequence[str],
                   slicing,
                   store: str,
                   compression: str,
                   compression_level: int,
                   overwrite: bool,
                   chunksize: Tuple[int],
                   method: str,
                   num_iterations: int,
                   max_correction: int,
                   power: float,
                   blind_spot: int,
                   back_projection: str,
                   objective: str,
                   numerical_aperture: float,
                   dxy: float,
                   dz: float,
                   xy_size: int,
                   z_size: int,
                   show_psf: bool,
                   scaling: Tuple[float],
                   workers: int,
                   workersbackend: str,
                   devices: List[int],
                   check: bool,
                   stop_at_exception: bool = True,
                   ):
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(path, mode, store)

    sz, sy, sx = scaling
    aprint(f"Input images will be scaled by: (sz,sy,sx)={scaling}")

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=devices)
    client = Client(cluster)
    aprint('Dask Client', client)

    lazy_computation = []

    for channel in dataset._selected_channels(channels):

        shape = dataset.shape(channel)
        array = dataset.get_array(channel)

        total_time_points = shape[0]
        time_points = list(range(total_time_points))
        if slicing is not None:
            aprint(f"Slicing with: {slicing}")
            if isinstance(slicing, tuple):
                time_points = time_points[slicing[0]]
                slicing = slicing[1:]
            else:  # slicing only over time
                time_points = time_points[slicing]
                slicing = ...
        else:
            slicing = ...

        new_shape = (len(time_points), ) + array[0][slicing].shape
        new_shape = tuple(int(round(u*v)) for u, v in zip(new_shape, (1,)+scaling))
        chunks = (1, 250, *shape[2:])  # trying to minimize the number of chunks per stack

        dest_array = dest_dataset.add_channel(name=channel,
                                              shape=new_shape,
                                              dtype=array.dtype,
                                              chunks=chunks,
                                              codec=compression,
                                              clevel=compression_level)

        #This is not ideal but difficult to avoid right now:
        sxy = (sx+sy)/2

        psf_kwargs = {'dxy': dxy / sxy,
                      'dz': dz / sz,
                      'xy_size': int(round(xy_size * sxy)),
                      'z_size': int(round(z_size * sz))}

        aprint(f"psf_kwargs: {psf_kwargs}")

        if numerical_aperture is not None:
            aprint(f"Numerical aperture overridden to a value of: {numerical_aperture}")
            psf_kwargs['NA'] = numerical_aperture

        if objective == 'nikon16x08na':
            psf_kernel = nikon16x08na(**psf_kwargs)
        elif objective == 'olympus20x10na':
            psf_kernel = olympus20x10na(**psf_kwargs)
        else:
            raise NotImplementedError

        if show_psf:
            from napari import gui_qt, Viewer
            with gui_qt():
                viewer = Viewer(title=f"DEXP | viewing PSF with napari", ndisplay=3)
                viewer.add_image(psf_kernel)

        @dask.delayed
        def process(i):
            print('BEGIN')
            tp = time_points[i]
            try:
                with asection(f"Loading channel: {channel} for time point {i}/{len(time_points)}"):
                    tp_array = numpy.asarray(array[tp][slicing])

                with BestBackend(exclusive=True, enable_unified_memory=True):

                    if sz != 1.0 or sy != 1.0 or sx != 1.0:
                        with asection(f"Applying scaling {(sz, sy, sx)} to image."):
                            sp = Backend.get_sp_module()
                            tp_array = Backend.to_backend(tp_array)
                            tp_array = sp.ndimage.interpolation.zoom(tp_array, zoom=(sz, sy, sx), order=1)
                            tp_array = Backend.to_numpy(tp_array)

                    if method == 'lr':
                        min_value = tp_array.min()
                        max_value = tp_array.max()

                        def f(image):
                            return lucy_richardson_deconvolution(image=image,
                                                                 psf=psf_kernel,
                                                                 num_iterations=num_iterations,
                                                                 max_correction=max_correction,
                                                                 normalise_minmax=(min_value, max_value),
                                                                 power=power,
                                                                 blind_spot=blind_spot,
                                                                 blind_spot_mode='median+uniform',
                                                                 blind_spot_axis_exclusion=(0,),
                                                                 back_projection=back_projection
                                                                 )

                        margins = max(xy_size, z_size)
                        with asection(f"LR Deconvolution of image of shape: {tp_array.shape}, with chunk size: {chunksize}, margins: {margins} "):
                            aprint(f"Number of iterations: {num_iterations}, back_projection:{back_projection}, ")
                            tp_array = scatter_gather_i2i(f, tp_array, chunks=chunksize, margins=margins)
                    else:
                        raise ValueError(f"Unknown deconvolution mode: {method}")

                    with asection(f"Moving array from backend to numpy."):
                        tp_array = Backend.to_numpy(tp_array, dtype=dest_array.dtype, force_copy=False)

                with asection(f"Saving deconvolved stack for time point {i}, shape:{tp_array.shape}, dtype:{array.dtype}"):
                    print('CHANNEL', channel)
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
            print('DONE')
            print(dest_dataset.info(channel))

        for i in range(len(time_points)):
            lazy_computation.append(process(i))

    dask.compute(*lazy_computation)

    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    dest_dataset.set_cli_history(parent=dataset if isinstance(dataset, ZDataset) else None)
    # close destination dataset:
    dest_dataset.close()
