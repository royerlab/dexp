from typing import Sequence, Optional

import numpy
from arbol.arbol import aprint
from skimage.transform import downscale_local_mean

from dexp.datasets.base_dataset import BaseDataset
from dexp.utils.timeit import timeit


def dataset_isonet(dataset: BaseDataset,
                   path: str,
                   channel: Optional[Sequence[str]],
                   slicing,
                   store: str,
                   compression: str,
                   compression_level: int,
                   overwrite: bool,
                   context: str,
                   mode: str,
                   dxy: float,
                   dz: float,
                   binning: int,
                   sharpening: bool,
                   training_tp_index: Optional[int],
                   max_epochs: int,
                   check: bool):
    if channel is None:
        channel = 'fused'

    if training_tp_index is None:
        training_tp_index = dataset.nb_timepoints(channel) // 2

    aprint(f"Selected channel {channel}")

    aprint(f"getting Dask arrays to apply isonet on...")
    array = dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True)

    if slicing is not None:
        aprint(f"Slicing with: {slicing}")
        array = array[slicing]

    aprint(f"Binning image by a factor {binning}...")
    dxy *= binning

    subsampling = dz / dxy
    aprint(f"Parameters: dxy={dxy}, dz={dz}, subsampling={subsampling}")

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
        dest_dataset = ZDataset(path, mode, store, parent=dataset)
        zarr_array = None

        for tp in range(0, array.shape[0] - 1):
            with timeit('Elapsed time: '):

                aprint(f"Processing time point: {tp} ...")
                tp_array = array[tp].compute()

                aprint("Downscaling image...")
                tp_array = downscale_local_mean(tp_array, factors=(1, binning, binning))
                # array_tp_downscaled = zoom(tp_array, zoom=(1, 1.0/binning, 1.0/binning), order=0)

                if sharpening:
                    aprint("Sharpening image...")
                    from dexp.processing.restoration import dehazing
                    tp_array = dehazing(tp_array, mode='hybrid', min=0, max=1024, margin_pad=False)

                aprint("Applying IsoNet to image...")
                tp_array = isonet.apply(tp_array)

                aprint(f'Result: image of shape: {tp_array.shape}, dtype: {tp_array.dtype} ')

                if zarr_array is None:
                    shape = (array.shape[0],) + tp_array.shape
                    aprint(f"Creating Zarr array of shape: {shape} ")
                    zarr_array = dest_dataset.add_channel(name=channel,
                                                          shape=shape,
                                                          dtype=array.dtype,
                                                          codec=compression,
                                                          clevel=compression_level)

                aprint(f'Writing image to Zarr file...')
                dest_dataset.write_stack(channel=channel,
                                         time_point=tp,
                                         stack_array=tp_array)

                aprint(f"Done processing time point: {tp}")

        aprint(dest_dataset.info())
        if check:
            dest_dataset.check_integrity()

        # close destination dataset:
        dest_dataset.close()
