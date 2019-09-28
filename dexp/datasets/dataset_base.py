import os
from abc import ABC, abstractmethod

import numpy
import zarr
from numcodecs.blosc import Blosc
from tifffile import memmap, TiffWriter
from zarr import open_group


class DatasetBase(ABC):

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
             project=None):

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
                print(f"Writing time point: {tp} ")

                tp_array = array[tp].compute()

                if project:
                    # project is the axis for projection, but here we are not considering the T dimension anymore...
                    axis = project - 1
                    tp_array = tp_array.max(axis=axis)

                z[tp] = tp_array

            from joblib import Parallel, delayed
            Parallel(n_jobs=-1)(delayed(process)(tp) for tp in range(0, shape[0] - 1))


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

        spx = shape[-1] // 2
        spz = shape[1] // 2
        smoothx = 60
        smoothz = 60

        print(f"Creating blend weights...")
        blending_x = numpy.fromfunction(lambda z, y, x: 1.0 - 0.5*(1+(((x-spx)/smoothx) / (1.0 + ((x-spx)/smoothx) ** 2) ** 0.5)), shape=shape[1:], dtype=numpy.float32)
        blending_z = numpy.fromfunction(lambda z, y, x: 1.0 - 0.5*(1+(((z-spz)/smoothz) / (1.0 + ((z-spz)/smoothz) ** 2) ** 0.5)), shape=shape[1:], dtype=numpy.float32)

        blending_C0L0 = blending_z*blending_x
        blending_C0L1 = blending_z*(1-blending_x)
        blending_C1L0 = (1-blending_z)*(blending_x)
        blending_C1L1 = (1-blending_z)*(1-blending_x)


        def process(tp):
            print(f"Writing time point: {tp} ")

            tp_array_C0L0 = array_C0L0[tp].compute()
            tp_array_C0L1 = array_C0L1[tp].compute()
            tp_array_C1L0 = numpy.flip(array_C1L0[tp].compute(),-1)
            tp_array_C1L1 = numpy.flip(array_C1L1[tp].compute(),-1)

            array    = blending_C0L0*tp_array_C0L0\
                       +blending_C0L1*tp_array_C0L1\
                       +blending_C1L0*tp_array_C1L0\
                       +blending_C1L1*tp_array_C1L1

            print(f'array dtype: {array.dtype}')

            zarr_array[tp] = array.astype(zarr_array.dtype)
            #zarr_array[tp] = blending_x.astype(zarr_array.dtype)

        #for tp in range(0, shape[0] - 1):
        #    process(tp)

        from joblib import Parallel, delayed
        Parallel(n_jobs=-1)(delayed(process)(tp) for tp in range(0, shape[0] - 1))


        print("Zarr tree:")
        print(root.tree())

        return root


    def tiff(self,
             path,
             channel,
             slice=None,
             overwrite=True):


        if not overwrite and os.path.exists(path):
            print(f"File {path} already exists! Set option -w to overwrite.")
            return

        print(f"getting Dask arrays for channel {channel}")
        array = self.get_stacks(channel, per_z_slice=False)


        if not slice is None:
            print(f"Slicing with: {slice}")
            array = array[slice]
            print(f"Done slicing.")


        shape = array.shape
        dtype = array.dtype

        print(f"Creating memory mapped TIFF file at: {path}.")
        with TiffWriter(path, bigtiff=True, imagej=True) as tif:
            for tp in range(0, shape[0] - 1):
                print(f"Writing time point: {tp} ")
                array_tp = array[tp].compute()
                tif.save(array_tp)


        # print(f"Creating memory mapped TIFF file at: {path}.")
        # memmap_tiff = memmap(path, shape=shape, dtype=dtype)
        #
        # def process(tp):
        #     print(f"Writing time point: {tp} ")
        #
        #     array_tp = array[tp].compute()
        #
        #     memmap_tiff[tp] = array_tp
        #
        # from joblib import Parallel, delayed
        # Parallel(n_jobs=-1)(delayed(process)(tp) for tp in range(0, shape[0] - 1))
        #
        # print(f"Flushing memory mapped TIFF file.")
        # memmap_tiff.flush()
        #
        # print(f"Flushing memory mapped TIFF file.")
        # del memmap_tiff


