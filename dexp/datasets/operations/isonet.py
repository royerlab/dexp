import numpy
from skimage.transform import downscale_local_mean

from dexp.utils.timeit import timeit


def dataset_isonet(dataset,
                   path,
                   channel,
                   slicing,
                   store,
                   compression,
                   compression_level,
                   overwrite,
                   context,
                   mode,
                   dxy,
                   dz,
                   binning,
                   sharpening,
                   training_tp_index,
                   max_epochs,
                   check):
    if channel is None:
        channel = 'fused'

    if training_tp_index is None:
        training_tp_index = dataset.nb_timepoints(channel) // 2

    print(f"Selected channel {channel}")

    print(f"getting Dask arrays to apply isonet on...")
    array = dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

    if slicing is not None:
        print(f"Slicing with: {slicing}")
        array = array[slicing]

    print(f"Binning image by a factor {binning}...")
    dxy *= binning

    subsampling = dz / dxy
    print(f"Parameters: dxy={dxy}, dz={dz}, subsampling={subsampling}")

    psf = numpy.ones((1, 1)) / 1
    print(f"PSF (along xy): {psf}")
    from dexp.processing.isonet import IsoNet
    isonet = IsoNet(context, subsampling=subsampling)

    if 'p' in mode:
        training_array_tp = array[training_tp_index].compute()
        training_downscaled_array_tp = downscale_local_mean(training_array_tp, factors=(1, binning, binning))
        print(f"Training image shape: {training_downscaled_array_tp.shape} ")
        isonet.prepare(training_downscaled_array_tp, psf=psf, threshold=0.999)

    if 't' in mode:
        isonet.train(max_epochs=max_epochs)

    if 'a' in mode:
        from dexp.datasets.zarr_dataset import ZDataset
        mode = 'w' + ('' if overwrite else '-')
        dest_dataset = ZDataset(path, mode, store)
        zarr_array = None

        for tp in range(0, array.shape[0] - 1):
            with timeit('Elapsed time: '):

                print(f"Processing time point: {tp} ...")
                array_tp = array[tp].compute()

                print("Downscaling image...")
                array_tp = downscale_local_mean(array_tp, factors=(1, binning, binning))
                # array_tp_downscaled = zoom(array_tp, zoom=(1, 1.0/binning, 1.0/binning), order=0)

                if sharpening:
                    print("Sharpening image...")
                    from dexp.processing.restoration import dehazing
                    array_tp = dehazing(array_tp, mode='hybrid', min=0, max=1024, margin_pad=False)

                print("Applying IsoNet to image...")
                array_tp = isonet.apply(array_tp)

                print(f'Result: image of shape: {array_tp.shape}, dtype: {array_tp.dtype} ')

                if zarr_array is None:
                    shape = (array.shape[0],) + array_tp.shape
                    print(f"Creating Zarr array of shape: {shape} ")
                    zarr_array = dest_dataset.add_channel(name=channel,
                                                          shape=shape,
                                                          dtype=array.dtype,
                                                          chunks=dataset._default_chunks,
                                                          codec=compression,
                                                          clevel=compression_level)

                print(f'Writing image to Zarr file...')
                zarr_array[tp] = array_tp.astype(zarr_array.dtype, copy=False)

                print(f"Done processing time point: {tp}")

        print(dest_dataset.info())
        if check:
            dest_dataset.check_integrity()
        dest_dataset.close()
