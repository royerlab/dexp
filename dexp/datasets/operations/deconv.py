from typing import Sequence, Tuple

from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean

from dexp.datasets.base_dataset import BaseDataset
from dexp.optics.psf.standard_psfs import nikon16x08na, olympus20x10na
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
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
                   dxy: float,
                   dz: float,
                   xy_size: int,
                   z_size: int,
                   downscalexy2: bool,
                   workers: int,
                   workersbackend: str,
                   devices: Sequence[int],
                   check: bool,
                   stop_at_exception: bool=True):

    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(path, mode, store)

    for channel in dataset._selected_channels(channels):

        array = dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

        if slicing is not None:
            array = array[slicing]

        shape = array.shape
        chunks = dataset._default_chunks

        dest_array = dest_dataset.add_channel(name=channel,
                                              shape=shape,
                                              dtype=array.dtype,
                                              chunks=chunks,
                                              codec=compression,
                                              clevel=compression_level)

        psf_kwargs = {'dxy': dxy * (2 if downscalexy2 else 1),
                      'dz': dz,
                      'xy_size': xy_size,
                      'z_size': z_size}

        aprint(f"PSF parameters: {psf_kwargs}")

        if objective == 'nikon16x08na':
            psf_kernel = nikon16x08na(**psf_kwargs)
        elif objective == 'olympus20x10na':
            psf_kernel = olympus20x10na(**psf_kwargs)

        def process(tp, device):

            try:
                with asection(f"Loading channel: {channel} for time point {tp}"):
                    tp_array = array[tp].compute()

                with CupyBackend(device, exclusive=True):

                    if downscalexy2:
                        tp_array = downscale_local_mean(tp_array, factors=(1, 2, 2)).astype(tp_array.dtype)

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

                with asection(f"Saving deconvolved stack for time point {tp}, shape:{array.shape}, dtype:{array.dtype}"):
                    dest_dataset.write_stack(channel=channel,
                                             time_point=tp,
                                             stack_array=tp_array)

                aprint(f"Done processing time point: {tp} .")

            except Exception as error:
                aprint(error)
                aprint(f"Error occurred while processing time point {tp} !")
                import traceback
                traceback.print_exc()

                if stop_at_exception:
                    raise error

        if workers == -1:
            workers = len(devices)
        aprint(f"Number of workers: {workers}")

        if workers > 1:
            Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp, devices[tp % len(devices)]) for tp in range(0, shape[0]))
        else:
            for tp in range(0, shape[0]):
                process(tp, devices[0])
    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
