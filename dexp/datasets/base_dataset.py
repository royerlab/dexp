import os
from abc import ABC, abstractmethod
from os.path import join

import numpy
import zarr
from numcodecs.blosc import Blosc
from scipy.ndimage import zoom
from skimage.transform import downscale_local_mean
from tifffile import memmap, TiffWriter
from zarr import open_group

from dexp.enhance.sharpen import sharpen
from dexp.fusion.fast_fusion import FastFusion
from dexp.io.io import tiff_save
from dexp.isonet.isonet import IsoNet
from dexp.utils.timeit import timeit


class BaseDataset(ABC):

    def __init__(self, dask_backed=False):
        """

        """

        self.dask_backed = dask_backed

    @abstractmethod
    def channels(self):
        pass

    @abstractmethod
    def shape(self, channel):
        pass

    @abstractmethod
    def nb_timepoints(self, channel):
        pass

    @abstractmethod
    def get_stacks(self, channel, per_z_slice):
        pass

    @abstractmethod
    def get_stack(self, channel, time_point, per_z_slice):
        pass

    def copy(self,
             path,
             channels=None,
             slice=None,
             compression='zstd',
             compression_level=3,
             chunk_size=512,
             chunk_all_z=False,
             overwrite=False,
             project=None,
             enhancements=[]):

        mode = 'w' + ('' if overwrite else '-')
        root = None
        try:
            root = open_group(path, mode=mode)
        except Exception as e:
            print(f"Problem: can't create target file/directory, most likely the target dataset already exists or path incorrect: {path}")
            return None

        filters = []  # [Delta(dtype='i4')]
        compressor = Blosc(cname=compression, clevel=compression_level, shuffle=Blosc.BITSHUFFLE)

        if channels is None:
            selected_channels = self._channels
        else:
            selected_channels = list(set(channels) & set(self._channels))

        print(f"Available channels: {self._channels}")
        print(f"Requested channels: {channels if channels else '--All--'} ")
        print(f"Selected channels:  {selected_channels}")

        for channel in selected_channels:

            channel_group = root.create_group(channel)

            array = self.get_stacks(channel, per_z_slice=False)

            if not slice is None:
                array = array[slice]

            if project:
                shape = array.shape[0:project] + array.shape[project + 1:]
                dim = len(shape)
                chunks = (1,) + (None,) * (dim - 1)
                print(f"projecting along axis {project} to shape: {shape} and chunks: {chunks}")

            else:
                shape = array.shape
                dim = len(shape)

                if dim <= 3:
                    chunks = (chunk_size,) * dim
                else:
                    if chunk_all_z:
                        chunks = (1,) * (dim - 3) + (chunk_size,) * 3
                    else:
                        chunks = (1,) * (dim - 2) + (chunk_size,) * 2


            print(f"Writing Zarr array for channel '{channel}' of shape {array.shape} ")

            z = channel_group.create(name=channel,
                                     shape=shape,
                                     dtype=array.dtype,
                                     chunks=chunks,
                                     filters=filters,
                                     compressor=compressor)

            def process(tp):
                print(f"Starting to process time point: {tp} ...")

                tp_array = array[tp].compute()

                for enhancement in enhancements:
                    print(f"Applying enhancement: {enhancement}...")
                    if enhancement.strip() == 'sharpen':
                        tp_array = sharpen(tp_array)
                    print(f"Done applying enhancement: {enhancement}")

                if project:
                    # project is the axis for projection, but here we are not considering the T dimension anymore...
                    axis = project - 1
                    tp_array = tp_array.max(axis=axis)

                z[tp] = tp_array
                print(f"Done processing time point: {tp} .")

            from joblib import Parallel, delayed

            number_of_workers = 1 if len(enhancements)>0 else -1

            print(f"Number of workers: {number_of_workers}")

            Parallel(n_jobs=number_of_workers)(delayed(process)(tp) for tp in range(0, shape[0] - 1))


        # print(root.info)
        print("Zarr tree:")
        print(root.tree())

        return root





    def fuse(self,
             path,
             slice=None,
             compression='zstd',
             compression_level=3,
             overwrite=False):

        print(f"getting Dask arrays for all channels to fuse...")
        array_C0L0 = self.get_stacks('C0L0', per_z_slice=False)
        array_C0L1 = self.get_stacks('C0L1', per_z_slice=False)
        array_C1L0 = self.get_stacks('C1L0', per_z_slice=False)
        array_C1L1 = self.get_stacks('C1L1', per_z_slice=False)

        if not slice is None:
            print(f"Slicing with: {slice}")
            array_C0L0 = array_C0L0[slice]
            array_C0L1 = array_C0L1[slice]
            array_C1L0 = array_C1L0[slice]
            array_C1L1 = array_C1L1[slice]

        shape = array_C0L0.shape

        mode = 'w' + ('' if overwrite else '-')
        root = None
        try:
            print(f"opening Zarr file for writing at: {path}")
            root = open_group(path, mode=mode)
        except Exception as e:
            print(
                f"Problem: can't create target file/directory, most likely the target dataset already exists or path incorrect.")
            return None

        filters = []  # [Delta(dtype='i4')]
        compressor = Blosc(cname=compression, clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
        channel_group = root.create_group('fused')
        print(f"Writing Zarr array for fused channel of shape {shape} ")
        zarr_array = channel_group.create(  name='fused',
                                            shape=shape,
                                            dtype=array_C0L0.dtype,
                                            chunks=(1, 1, 512, 512),
                                            filters=filters,
                                            compressor=compressor)

        fusion = FastFusion(zarr_array[0].shape)

        def process(tp):
            print(f"Writing time point: {tp} ")

            C0L0 = array_C0L0[tp].compute()
            C0L1 = array_C0L1[tp].compute()
            C1L0 = numpy.flip(array_C1L0[tp].compute(),-1)
            C1L1 = numpy.flip(array_C1L1[tp].compute(),-1)

            array = fusion.fuse(C0L0, C0L1, C1L0, C1L1)

            print(f'array dtype: {array.dtype}')

            zarr_array[tp] = array.astype(zarr_array.dtype)


        from joblib import Parallel, delayed
        Parallel(n_jobs=-1)(delayed(process)(tp) for tp in range(0, shape[0] - 1))


        print("Zarr tree:")
        print(root.tree())

        return root

    def isonet(self,
                 path,
                 channel=None,
                 slice=None,
                 compression='zstd',
                 compression_level=3,
                 overwrite=False,
                 context='default',
                 mode='pta',
                 dxy = 0.4,
                 dz  = 6.349,
                 binning = 2,
                 sharpening = False,
                 training_tp_index=0,
                 max_epochs=100):


        if channel is None:
            channel = 'fused'

        print(f"Selected channel {channel}")

        print(f"getting Dask arrays to apply isonet on...")
        array = self.get_stacks(channel, per_z_slice=False)

        if not slice is None:
            print(f"Slicing with: {slice}")
            array = array[slice]

        print(f"Binning image by a factor {binning}...")
        dxy *= binning

        subsampling = dz / dxy
        print(f"Parameters: dxy={dxy}, dz={dz}, subsampling={subsampling}")


        isonet = IsoNet(context)

        if 'p' in mode:
            psf = numpy.ones((1, 1)) / 1

            training_array_tp = array[training_tp_index].compute()
            isonet.prepare(training_array_tp, subsampling=subsampling, psf=psf, threshold=0.99)

        if 't' in mode:
            isonet.train(max_epochs=max_epochs)

        if 'a' in mode:

            mode = 'w' + ('' if overwrite else '-')
            root = None
            try:
                print(f"opening Zarr file for writing at: {path}")
                root = open_group(path, mode=mode)
            except Exception as e:
                print(
                    f"Problem: can't create target file/directory, most likely the target dataset already exists or path incorrect.")
                return None

            filters = []  # [Delta(dtype='i4')]
            compressor = Blosc(cname=compression, clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
            channel_group = root.create_group(channel)

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
                        array_tp = sharpen(array_tp, mode='hybrid', min=0, max=1024, margin_pad=False)

                    print("Applying IsoNet to image...")
                    array_tp = isonet.apply(array_tp)

                    print(f'Result: image of shape: {array_tp.shape}, dtype: {array_tp.dtype} ')

                    if zarr_array is None:
                        shape = (array.shape[0],)+array_tp.shape
                        print(f"Creating Zarr array of shape: {shape} ")
                        zarr_array = channel_group.create(name=channel,
                                                          shape=shape,
                                                          dtype=array.dtype,
                                                          chunks=(1, 1, 512, 512),
                                                          filters=filters,
                                                          compressor=compressor)

                    print(f'Writing image to Zarr file...')
                    zarr_array[tp] = array_tp.astype(zarr_array.dtype, copy=False)

                    print(f"Done processing time point: {tp}")



            print("Zarr tree:")
            print(root.tree())

            return root





    def tiff(self,
             path,
             channel,
             slice=None,
             overwrite=True,
             one_file_per_first_dim=True):


        if not overwrite and os.path.exists(path):
            print(f"File {path} already exists! Set option -w to overwrite.")
            return

        print(f"getting Dask arrays for channel {channel}")
        array = self.get_stacks(channel, per_z_slice=False)


        if not slice is None:
            print(f"Slicing with: {slice}")
            array = array[slice]
            print(f"Done slicing.")


        if one_file_per_first_dim:
            print(f"Saving one TIFF file for each tp (or Z if already sliced) to: {path}.")

            os.makedirs(path, exist_ok=True)

            tp=0
            for stack in array:
                print(f"Writing time point: {tp} ")
                tiff_save(join(path, f"file{tp}.tiff"), stack.compute())
                tp+=1


        else:
            print(f"Creating memory mapped TIFF file at: {path}.")
            with TiffWriter(path, bigtiff=True, imagej=True) as tif:
                tp = 0
                for stack in array:
                    print(f"Writing time point: {tp} ")
                    stack = stack.compute()
                    tif.save(stack)
                    tp += 1

