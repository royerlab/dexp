import os
from abc import ABC, abstractmethod
from os.path import join

import numpy
from numcodecs import blosc
from numcodecs.blosc import Blosc

from skimage.transform import downscale_local_mean
from tifffile import TiffWriter
from zarr import open_group, ZipStore, DirectoryStore, convenience

from dexp.io.io import tiff_save
from dexp.utils.timeit import timeit


class BaseDataset(ABC):

    def __init__(self, dask_backed=False):
        """

        """

        self.dask_backed = dask_backed

        self._chunk_xy = 512
        self._chunk_z  = 32
        self._chunk_t  = 1

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
             channels,
             slicing,
             store,
             compression,
             compression_level,
             overwrite,
             project,
             workers):

        mode = 'w' + ('' if overwrite else '-')
        try:
            if store == 'zip':
                path = path if path.endswith('.zip') else path+'.zip'
                store = ZipStore(path)
            elif  store == 'dir':
                store = DirectoryStore(path)
            root = open_group(store, mode=mode)
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

            array = self.get_stacks(channel, per_z_slice=False)

            if slicing is not None:
                array = array[slicing]

            if project:
                shape = array.shape[0:project] + array.shape[project + 1:]
                dim = len(shape)
                chunks = (1,) + (None,) * (dim - 1)
                print(f"projecting along axis {project} to shape: {shape} and chunks: {chunks}")

            else:
                shape = array.shape
                dim = len(shape)

                if dim == 3:
                        chunks = (self._chunk_z, self._chunk_xy, self._chunk_xy)
                elif dim == 4:
                        chunks = (self._chunk_t, self._chunk_z, self._chunk_xy, self._chunk_xy)


            print(f"Writing Zarr array for channel '{channel}' of shape {shape} ")
            channel_group = root.create_group(channel)
            z = channel_group.create(name=channel,
                                     shape=shape,
                                     dtype=array.dtype,
                                     chunks=chunks,
                                     filters=filters,
                                     compressor=compressor)


            def process(tp):

                try:

                    print(f"Starting to process time point: {tp} ...")

                    tp_array = array[tp].compute()

                    if project:
                        # project is the axis for projection, but here we are not considering the T dimension anymore...
                        axis = project - 1
                        tp_array = tp_array.max(axis=axis)

                    z[tp] = tp_array
                    print(f"Done processing time point: {tp} .")

                except Exception as error:
                    print(error)
                    print(f"Error occurred while copying time point {tp} !")


            from joblib import Parallel, delayed

            if workers is None:
                workers = os.cpu_count()//2

            print(f"Number of workers: {workers}")

            Parallel(n_jobs=workers)(delayed(process)(tp) for tp in range(0, shape[0]))


        # print(root.info)
        print("Zarr tree:")
        print(root.tree())

        store.close()



    def fuse(self,
             path,
             slicing,
             store,
             compression,
             compression_level,
             overwrite,
             workers,
             zero_level,
             load_shifts):

        print(f"getting Dask arrays for all channels to fuse...")
        array_C0L0 = self.get_stacks('C0L0', per_z_slice=False)
        array_C0L1 = self.get_stacks('C0L1', per_z_slice=False)
        array_C1L0 = self.get_stacks('C1L0', per_z_slice=False)
        array_C1L1 = self.get_stacks('C1L1', per_z_slice=False)

        if slicing is not None:
            print(f"Slicing with: {slicing}")
            array_C0L0 = array_C0L0[slicing]
            array_C0L1 = array_C0L1[slicing]
            array_C1L0 = array_C1L0[slicing]
            array_C1L1 = array_C1L1[slicing]

        shape = array_C0L0.shape

        mode = 'w' + ('' if overwrite else '-')
        root = None
        try:
            print(f"opening Zarr file for writing at: {path}")
            if store == 'zip':
                path = path if path.endswith('.zip') else path+'.zip'
                store = ZipStore(path)
            elif  store == 'dir':
                store = DirectoryStore(path)
            root = open_group(store, mode=mode)
        except Exception as e:
            print(
                f"Problem: can't create target file/directory, most likely the target dataset already exists or path incorrect.")
            return None

        dtype = array_C0L0.dtype
        filters = []
        compressor = Blosc(cname=compression, clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
        channel_group = root.create_group('fused')
        print(f"Writing Zarr array for fused channel of shape {shape} ")
        zarr_array = channel_group.create(name='fused',
                                          shape=shape,
                                          dtype=dtype,
                                          chunks=(self._chunk_t, self._chunk_z, self._chunk_xy, self._chunk_xy),
                                          filters=filters,
                                          compressor=compressor)

        from dexp.fusion.fusion import SimpleFusion
        fusion = SimpleFusion()

        shifts_file = open("registration_shifts.txt", "r" if load_shifts else 'w')
        if load_shifts:
            print(f"Loading registration shifts from existing file! ({shifts_file.name})")

        def process(tp):
            print(f"Writing time point: {tp} ")

            C0L0 = array_C0L0[tp].compute()
            C0L1 = array_C0L1[tp].compute()
            C1L0 = array_C1L0[tp].compute()
            C1L1 = array_C1L1[tp].compute()

            C1L0 = numpy.flip(C1L0, -1)
            C1L1 = numpy.flip(C1L1, -1)

            if load_shifts:
                try:
                    line = shifts_file.readline().strip()
                    shifts = tuple(float(shift) for shift in line.split('\t'))
                    print(f"loaded shifts: {shifts} ")
                except ValueError:
                    print(f"Cannot read shift from line: {line}, most likely we have reached the end of the shifts file, have the channels a different number of time points?")

            else:
                shifts = None

            print(f'Fusing...')
            array, shifts = fusion.fuse_2I2D(C0L0, C0L1, C1L0, C1L1, shifts=shifts, zero_level=zero_level, as_numpy=True)

            if not load_shifts:
                for shift in shifts:
                    shifts_file.write(f"{shift}\t")
                shifts_file.write(f"\n")

            array = array.astype(zarr_array.dtype, copy=False)

            print(f'Writing array of dtype: {array.dtype}')
            zarr_array[tp] = array

        if workers is None:
            workers = os.cpu_count()//2

        if workers == 1:
            for tp in range(0, shape[0]):
                process(tp)
        else:
            from joblib import Parallel, delayed
            Parallel(n_jobs=workers, backend='threading')(delayed(process)(tp) for tp in range(0, shape[0]))

        print("Zarr tree:")
        print(root.tree())

        shifts_file.close()

        store.close()


    def deconv(self,
               path,
               channels,
               slicing,
               store,
               compression,
               compression_level,
               overwrite,
               workers,
               method,
               num_iterations,
               max_correction,
               power,
               dxy,
               dz,
               xy_size,
               z_size,
               downscalexy2):

        mode = 'w' + ('' if overwrite else '-')
        root = None
        try:
            if store == 'zip':
                path = path if path.endswith('.zip') else path+'.zip'
                store = ZipStore(path)
            elif  store == 'dir':
                store = DirectoryStore(path)
            root = open_group(store, mode=mode)
        except Exception as e:
            print(f"Problem: can't create target file/directory, most likely the target dataset already exists or path incorrect: {path}")
            return None

        filters = []
        compressor = Blosc(cname=compression, clevel=compression_level, shuffle=Blosc.BITSHUFFLE)

        if channels is None:
            selected_channels = self._channels
        else:
            selected_channels = list(set(channels) & set(self._channels))

        print(f"Available channels: {self._channels}")
        print(f"Requested channels: {channels if channels else '--All--'} ")
        print(f"Selected channels:  {selected_channels}")


        for channel in selected_channels:

            array = self.get_stacks(channel, per_z_slice=False)

            if slicing is not None:
                array = array[slicing]

            shape = array.shape
            dim = len(shape)

            if dim == 3:
                chunks = (self._chunk_z, self._chunk_xy, self._chunk_xy)
            elif dim == 4:
                chunks = (self._chunk_t, self._chunk_z, self._chunk_xy, self._chunk_xy)


            print(f"Writing Zarr array for channel '{channel}' of shape {shape} ")
            channel_group = root.create_group(channel)
            z = channel_group.create(name=channel,
                                     shape=shape,
                                     dtype=array.dtype,
                                     chunks=chunks,
                                     filters=filters,
                                     compressor=compressor)

            from dexp.restoration.deconvolution import Deconvolution
            deconvolution = Deconvolution(method=method,
                                          num_iterations=num_iterations,
                                          max_correction=max_correction,
                                          power=power,
                                          dxy=dxy * (2 if downscalexy2 else 1),
                                          dz=dz,
                                          xy_size=xy_size,
                                          z_size=z_size)

            def process(tp):

                try:

                    print(f"Starting to process time point: {tp} ...")

                    tp_array = array[tp].compute()

                    if downscalexy2:
                        tp_array = downscale_local_mean(tp_array, factors=(1, 2, 2)).astype(tp_array.dtype)

                    tp_array = deconvolution.restore(tp_array)

                    z[tp] = tp_array
                    print(f"Done processing time point: {tp} .")

                except Exception as error:
                    print(error)
                    print(f"Error occurred while copying time point {tp} !")
                    import traceback
                    traceback.print_exc()


            from joblib import Parallel, delayed

            if workers is None:
                workers = os.cpu_count()//2

            print(f"Number of workers: {workers}")

            Parallel(n_jobs=workers)(delayed(process)(tp) for tp in range(0, shape[0]))


        # print(root.info)
        print("Zarr tree:")
        print(root.tree())

        store.close()



    def isonet(self,
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
               max_epochs):



        if channel is None:
            channel = 'fused'

        if training_tp_index is None:
            training_tp_index = self.nb_timepoints(channel)//2

        print(f"Selected channel {channel}")

        print(f"getting Dask arrays to apply isonet on...")
        array = self.get_stacks(channel, per_z_slice=False)

        if slicing is not None:
            print(f"Slicing with: {slicing}")
            array = array[slicing]

        print(f"Binning image by a factor {binning}...")
        dxy *= binning

        subsampling = dz / dxy
        print(f"Parameters: dxy={dxy}, dz={dz}, subsampling={subsampling}")

        psf = numpy.ones((1, 1)) / 1
        print(f"PSF (along xy): {psf}")
        from dexp.isonet.isonet import IsoNet
        isonet = IsoNet(context, subsampling=subsampling)


        if 'p' in mode:

            training_array_tp = array[training_tp_index].compute()

            training_downscaled_array_tp = downscale_local_mean(training_array_tp, factors=(1, binning, binning))

            print(f"Training image shape: {training_downscaled_array_tp.shape} ")

            isonet.prepare(training_downscaled_array_tp, psf=psf, threshold=0.999)

        if 't' in mode:
            isonet.train(max_epochs=max_epochs)

        if 'a' in mode:

            mode = 'w' + ('' if overwrite else '-')
            root = None
            try:
                print(f"opening Zarr file for writing at: {path}")
                if store == 'zip':
                    path = path if path.endswith('.zip') else path+'.zip'
                    store = ZipStore(path)
                elif store == 'dir':
                    store = DirectoryStore(path)
                root = open_group(store, mode=mode)
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
                        from dexp.restoration.sharpen import sharpen
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
                                                          chunks=(self._chunk_t, self._chunk_z, self._chunk_xy, self._chunk_xy),
                                                          filters=filters,
                                                          compressor=compressor)

                    print(f'Writing image to Zarr file...')
                    zarr_array[tp] = array_tp.astype(zarr_array.dtype, copy=False)

                    print(f"Done processing time point: {tp}")

            print("Zarr tree:")
            print(root.tree())

            store.close()





    def tiff(self,
             path,
             channels,
             slicing,
             overwrite,
             project,
             one_file_per_first_dim,
             clevel,
             workers):

        if channels is None:
            selected_channels = self._channels
        else:
            selected_channels = list(set(channels) & set(self._channels))

        print(f"Available channels: {self._channels}")
        print(f"Requested channels: {channels if channels else '--All--'} ")
        print(f"Selected channels:  {selected_channels}")

        print(f"getting Dask arrays for channels {selected_channels}")
        arrays = list([self.get_stacks(channel, per_z_slice=False) for channel in selected_channels])

        if slicing is not None:
            print(f"Slicing with: {slicing}")
            arrays = list([array[slicing] for array in arrays])
            print(f"Done slicing.")

        if project:
            # project is the axis for projection, but here we are not considering the T dimension anymore...
            print(f"Projecting along axis {project}")
            arrays = list([array.max(axis=project) for array in arrays])


        if one_file_per_first_dim:
            print(f"Saving one TIFF file for each tp (or Z if already sliced) to: {path}.")

            os.makedirs(path, exist_ok=True)

            from joblib import Parallel, delayed

            def process(tp):
                with timeit('Elapsed time: '):
                    for channel, array in zip(selected_channels, arrays):
                        tiff_file_path = join(path, f"file{tp}_{channel}.tiff")
                        if overwrite or not os.path.exists(tiff_file_path):
                            stack = array[tp].compute()
                            print(f"Writing time point: {tp} of shape: {stack.shape}, dtype:{stack.dtype} as TIFF file: '{tiff_file_path}', with compression: {clevel}")
                            tiff_save(tiff_file_path, stack, compress=clevel)
                            print(f"Done writing time point: {tp} !")
                        else:
                            print(f"File for time point (or z slice): {tp} already exists.")

            Parallel(n_jobs=workers)(delayed(process)(tp) for tp in range(0, arrays[0].shape[0]))


        else:
            array = numpy.stack(arrays)

            if not overwrite and os.path.exists(path):
                print(f"File {path} already exists! Set option -w to overwrite.")
                return

            print(f"Creating memory mapped TIFF file at: {path}.")
            with TiffWriter(path, bigtiff=True, imagej=True) as tif:
                tp = 0
                for stack in array:
                    with timeit('Elapsed time: '):
                        print(f"Writing time point: {tp} ")
                        stack = stack.compute()
                        tif.save(stack)
                        tp += 1

