from skimage.transform import downscale_local_mean

from dexp.optics.psf.simple_microscope_psf import SimpleMicroscopePSF
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.utils.scatter_gather import scatter_gather
from dexp.utils.timeit import timeit


def dataset_deconv(dataset,
                   path,
                   channels,
                   slicing,
                   store,
                   compression,
                   compression_level,
                   overwrite,
                   workers,
                   chunksize,
                   method,
                   num_iterations,
                   max_correction,
                   power,
                   dxy,
                   dz,
                   xy_size,
                   z_size,
                   downscalexy2,
                   check):
    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(path, mode, store)

    for channel in dataset._selected_channels(channels):

        array = dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

        if slicing is not None:
            array = array[slicing]

        shape = array.shape
        dim = len(shape)

        if dim == 3:
            chunks = dataset._default_chunks[1:]
        elif dim == 4:
            chunks = dataset._default_chunks

        dest_array = dest_dataset.add_channel(name=channel,
                                              shape=shape,
                                              dtype=array.dtype,
                                              chunks=chunks,
                                              codec=compression,
                                              clevel=compression_level)

        psf = SimpleMicroscopePSF()
        psf_kernel = psf.generate_xyz_psf(dxy=dxy * (2 if downscalexy2 else 1),
                                          dz=dz,
                                          xy_size=xy_size,
                                          z_size=z_size)
        psf_kernel /= psf_kernel.sum()

        def process(tp):

            backend = CupyBackend(0)

            try:
                print(f"Starting to process time point: {tp} ...")
                tp_array = array[tp].compute()
                if downscalexy2:
                    tp_array = downscale_local_mean(tp_array, factors=(1, 2, 2)).astype(tp_array.dtype)

                if method == 'lr':
                    def f(image):
                        return lucy_richardson_deconvolution(backend,
                                                             image=image,
                                                             psf=psf_kernel,
                                                             num_iterations=num_iterations,
                                                             max_correction=max_correction,
                                                             power=power)

                    with timeit("lucy_richardson_deconvolution"):
                        tp_array = scatter_gather(backend, f, tp_array, chunks=chunksize, margins=max(xy_size, z_size)//2)
                else:
                    raise ValueError(f"Unknown deconvolution mode: {method}")

                tp_array = backend.to_numpy(tp_array, dtype=dest_array.dtype, force_copy=False)
                dest_array[tp] = tp_array
                print(f"Done processing time point: {tp} .")

            except Exception as error:
                print(error)
                print(f"Error occurred while copying time point {tp} !")
                import traceback
                traceback.print_exc()

        # TODO: we are not yet distributing computation over GPUs, that would require a proper use of DASK for that.
        # See: https://medium.com/rapids-ai/parallelizing-custom-cupy-kernels-with-dask-4d2ccd3b0732

        for tp in range(0, shape[0]):
            process(tp)

    print(dest_dataset.info())
    if check:
        dest_dataset.check_integrity()
    dest_dataset.close()
