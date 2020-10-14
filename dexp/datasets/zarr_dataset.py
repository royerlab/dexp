import os
from multiprocessing.pool import ThreadPool
from os.path import isfile, isdir, exists
from typing import Tuple

import dask
import psutil
import zarr
from dask.array import from_zarr
from numcodecs import blosc
from zarr import ZipStore, DirectoryStore, open_group, convenience, CopyError, Blosc

from dexp.datasets.base_dataset import BaseDataset
import multiprocessing

# Configure multithreading for Dask:
nb_threads = multiprocessing.cpu_count()//2-1
dask.config.set(scheduler='threads')
dask.config.set(pool=ThreadPool(nb_threads))

# Configure multithreading for Blosc:
blosc.set_nthreads(nb_threads//2 - 1)

class ZDataset(BaseDataset):

    def __init__(self, path:str, mode:str ='r', store:str ='dir'):
        """Convenience function to open a group or array using file-mode-like semantics.

        Parameters
        ----------
        path : path to zarr styorage (directory or zip).
        mode : Access mode:
            'r' means read only (must exist);
            'r+' means read/write (must exist);
            'a' means read/write (create if doesn't exist);
            'w' means create (overwrite if exists);
            'w-' means create (fail if exists).
        store : type of store, can be 'dir' or 'zip'

        Returns
        -------
        Zarr dataset


        """


        super().__init__(dask_backed=False)

        self._folder = path

        if exists(path) and mode=='r':
            print(f"Storage exist but must be overwritten, deleting {path} !")
            os.remove(path)

        print(f"Initialising Zarr storage: '{path}'")
        if exists(path):
            print(f"Path exists, opening zarr storage...")
            self._root_group = zarr.convenience.open(path, mode=mode)
            self._initialise_existing(path)
            self._store = None
        else:
            try:
                print(f"Path does not exist, creating zarr storage...")
                if isfile(path) and (path.endswith('.zarr.zip') or store=='zip' ):
                    print(f"Opening as ZIP store")
                    # correcting path to adhere to convention:
                    path = path+'.zip' if path.endswith('.zarr') else path
                    path = path if path.endswith('.zarr.zip') else path+'zarr.zip'
                    self._store = zarr.storage.ZipStore(path)
                elif isdir(path) and ( path.endswith('.zarr') or path.endswith('.zarr/') or store=='dir'):
                    print(f"Opening as Nested Directory store")
                    # correcting path to adhere to convention:
                    path = path if path.endswith('.zarr') else path+'.zarr'
                    self._store = zarr.storage.NestedDirectoryStore(path)
                else:
                    print(f'Cannot open {path}, needs to be a zarr directory (directory that ends with `.zarr`), or a zipped zarr file (file that ends with `.zarr.zip`)')
                print(f"Opening Zarr storage with mode='{mode}'")
                self._root_group = zarr.convenience.open(store, mode=mode)
                self._initialise_empty()
            except Exception as e:
                print(f"Problem: can't create target file/directory, most likely the target dataset already exists or path incorrect: {path}")
                return None


    def _initialise_empty(self):
        self._channels = []
        self._arrays = {}

    def _initialise_existing(self, path):
        self._channels = [channel for channel, _ in self._root_group.groups()]
        self._arrays = {}

        print(f"Exploring Zarr hierarchy...")
        for channel, channel_group in self._root_group.groups():
            print(f"Found channel: {channel}")

            channel_items = channel_group.items()

            for item_name, array in channel_items:
                print(f"Found array: {item_name}")

                if item_name == channel or item_name == 'fused':
                    self._arrays[channel] = from_zarr(path, component=f"{channel}/{item_name}")

    def _get_group_for_channel(self, channel):
        groups = [g for c, g in self._root_group.groups() if c == channel]
        if len(groups) == 0:
            return None
        else:
            return groups[0]

    def close(self):
        # We close the store if it exists, i.e. if we have been writing to the dataset
        if self._store is not None:
            try:
                self._store.close()
            except AttributeError:
                pass

    def channels(self):
        return list(self._channels)

    def shape(self, channel):
        return self._arrays[channel].shape

    def chunks(self, channel):
        return self._arrays[channel].chunks

    def nb_timepoints(self, channel):
        return self.get_stacks(channel).shape[0]

    def tree(self):
        return self._root_group.tree()

    def info(self, channel=None):

        if channel:
            info_str = f"Channel: '{channel}'', nb time points: {self.nb_timepoints(channel)}, shape: {self.shape(channel)}"
            info_str += "\n"
            info_str += str(self._arrays[channel].info)
            return info_str
        else:
            info_str = str(self._root_group.tree())
            info_str += "\n\n"
            info_str += "Channels: \n"
            for channel in self.channels():
                info_str += "  └──" + self.info(channel) + "\n\n"

            return info_str

    def get_stacks(self, channel, per_z_slice=False):

        return dask.array.from_array(self._arrays[channel])

    def get_stack(self, channel, time_point, per_z_slice=False):

        stack_array = self.get_stacks(channel)[time_point]
        return stack_array


    def add_channel(self, name:str, shape:Tuple[int], dtype, chunks:Tuple[int], codec:str = 'zstd', clevel:int = 3):
        """Adds a channel to this dataset

        Parameters
        ----------
        name : name of channel.
        shape : shape of correspodning array.
        dtype : dtype of array.
        chunks: chunks shape.
        codec: Compression codec to be used ('zstd', 'blosclz', 'lz4', 'lz4hc', 'zlib' or 'snappy').
        clevel: An integer between 0 and 9 specifying the compression level.

        Returns
        -------
        zarr array


        """

        print(f"Adding channel: '{name}' of shape: {shape}, dtype: {dtype}, and chunks:{chunks} ")
        compressor = Blosc(cname=codec, clevel=clevel, shuffle=Blosc.BITSHUFFLE)
        filters = []

        channel_group = self._root_group.create_group(name)
        array = channel_group.create(name=name,
                                     shape=shape,
                                     dtype=dtype,
                                     chunks=chunks,
                                     filters=filters,
                                     compressor=compressor)

        return array



    def add_channels_to(self,
                        path,
                        channels,
                        rename,
                        store,
                        overwrite
                        ):
        """Adds channels from this zarr dataset into an other possibly existing zarr dataset

        Parameters
        ----------
        path : name of channel.
        channels: list or tuple of channels to add
        rename: list or tuple of new names for channels
        store: type of zarr store: 'dir' or 'zip'
        overwrite: overwrite destination (not fully functional for zip stores!)

        Returns
        -------
        zarr array


        """

        mode = 'w' + ('' if overwrite else '-')

        zdataset = ZDataset(path, mode, store)
        root = zdataset._root_group

        print(f"Existing channels: {zdataset.channels()}")

        for channel, new_name in zip(channels, rename):
            try:
                array = self.get_stacks(channel, per_z_slice=False)
                source_group = self._get_group_for_channel(channel)
                source_array = tuple((a for n,a in source_group.items() if n == channel))[0]

                print(f"Creating group for channel {channel} of new name {new_name}.")
                if new_name not in root.group_keys():
                    dest_group = root.create_group(new_name)
                else:
                    dest_group = root[new_name]

                print(f"Fast copying channel {channel} renamed to {new_name} of shape {array.shape} and dtype {array.dtype} ")
                convenience.copy(source_array, dest_group, if_exists='replace' if overwrite else 'raise')
            except CopyError:
                print(f"Channel already exists, set option '-w' to force overwriting! ")




