from typing import Sequence, Tuple

from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean

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
                   devices: Sequence[int],
                   check: bool,
                   stop_at_exception: bool = True,):
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(path, mode, store)

    sz, sy, sx = scaling
    aprint(f"Input images will be scaled by: (sz,sy,sx)={scaling}")

    for channel in dataset._selected_channels(channels):

        array = dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

        if slicing is not None:
            array = array[slicing]

        shape = tuple(int(round(u*v)) for u, v in zip(array.shape, (1,)+scaling))
        chunks = ZDataset._default_chunks
        nb_timepoints = shape[0]

        dest_array = dest_dataset.add_channel(name=channel,
                                              shape=shape,
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

        if show_psf:
            from napari import gui_qt, Viewer
            with gui_qt():
                viewer = Viewer(title=f"DEXP | viewing PSF with napari", ndisplay=3)
                viewer.add_image(psf_kernel)


        def process(tp, device):

            try:
                with asection(f"Loading channel: {channel} for time point {tp}/{nb_timepoints}"):
                    tp_array = array[tp].compute()

                with BestBackend(device, exclusive=True):

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

                with asection(f"Saving deconvolved stack for time point {tp}, shape:{tp_array.shape}, dtype:{array.dtype}"):
                    dest_dataset.write_stack(channel=channel,
                                             time_point=tp,
                                             stack_array=tp_array)

                aprint(f"Done processing time point: {tp}/{nb_timepoints} .")

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
            Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp, devices[tp % len(devices)]) for tp in range(0, nb_timepoints))
        else:
            for tp in range(0, nb_timepoints):
                process(tp, devices[0])


    aprint(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()

    dest_dataset.set_cli_history(parent=dataset if isinstance(dataset, ZDataset) else None)
    # close destination dataset:
    dest_dataset.close()
