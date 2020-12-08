import multiprocessing
import os
import shutil
from multiprocessing.pool import ThreadPool
from os.path import isfile, isdir, exists
from typing import Tuple, Sequence, Any, Union

import dask
import zarr
from arbol.arbol import aprint
from numcodecs import blosc
from zarr import open_group, convenience, CopyError, Blosc, Group

from dexp.datasets.base_dataset import BaseDataset

# Configure multithreading for Dask:
nb_threads = max(1, multiprocessing.cpu_count() // 2)
dask.config.set(scheduler='threads')
dask.config.set(pool=ThreadPool(nb_threads))

# Configure multithreading for Blosc:
blosc.set_nthreads(nb_threads // 2 - 1)


class ZDataset(BaseDataset):

    def __init__(self, path: str, mode: str = 'r', store: str = 'dir'):
        """Convenience function to open a group or array using file-mode-like semantics.

        Parameters
        ----------
        path : path to zarr storage (directory or zip).
        mode : Access mode:
            'r' means read only (must exist);
            'r+' means read/write (must exist);
            'a' means read/write (create if doesn't exist);
            'w' means create (overwrite if exists);
            'w-' means create (fail if exists).
        store : type of store, can be 'dir', 'ndir', or 'zip'

        Returns
        -------
        Zarr dataset


        """

        super().__init__(dask_backed=False)

        self._folder = path
        self._store = None
        self._root_group = None
        self._arrays = {}

        if exists(path) and mode == 'w-':
            raise ValueError(f"ERROR -- Storage '{path}' already exists, add option '-w' to force overwrite!")
        elif exists(path) and mode == 'w':
            aprint(f"Deleting '{path}' for overwrite!")
            if isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif isfile(path):
                os.remove(path)

        aprint(f"Initialising Zarr storage: '{path}' with read/write mode: '{mode}' and store type: '{store}'")
        if exists(path):
            aprint(f"Path exists, opening zarr storage...")
            if isfile(path) and (path.endswith('.zarr.zip') or store == 'zip'):
                aprint(f"Opening as ZIP store")
                self._store = zarr.storage.ZipStore(path)
            elif isdir(path) and ((path.endswith('.nested.zarr') or path.endswith('.nested.zarr/') or store == 'ndir')):
                aprint(f"Opening as Nested Directory store")
                self._store = zarr.storage.NestedDirectoryStore(path)
            elif isdir(path) and ((path.endswith('.zarr') or path.endswith('.zarr/') or store == 'dir')):
                aprint(f"Opening as Directory store")
                self._store = zarr.storage.DirectoryStore(path)

            aprint(f"Opening with mode: {mode}")
            self._root_group = open_group(self._store, mode=mode)
            self._initialise_existing(path)
        elif 'a' in mode or 'w' in mode:
            try:
                aprint(f"Path does not exist, creating zarr storage...")
                store_type = None
                if path.endswith('.zarr.zip') or store == 'zip':
                    aprint(f"Opening as ZIP store")
                    # correcting path to adhere to convention:
                    path = path + '.zip' if path.endswith('.zarr') else path
                    path = path if path.endswith('.zarr.zip') else path + 'zarr.zip'
                    self._store = zarr.storage.ZipStore(path)
                    store_type = 'zip'
                elif path.endswith('.nested.zarr') or path.endswith('.nested.zarr/') or store == 'ndir':
                    aprint(f"Opening as Nested Directory store")
                    # correcting path to adhere to convention:
                    path = path if path.endswith('.nested.zarr') else path + '.nested.zarr'
                    self._store = zarr.storage.NestedDirectoryStore(path)
                    store_type = 'ndir'
                elif path.endswith('.zarr') or path.endswith('.zarr/') or store == 'dir':
                    aprint(f"Opening as Directory store")
                    # correcting path to adhere to convention:
                    path = path if path.endswith('.zarr') else path + '.zarr'
                    self._store = zarr.storage.DirectoryStore(path)
                    store_type = 'dir'

                else:
                    aprint(f'Cannot open {path}, needs to be a zarr directory (directory that ends with `.zarr` or `.nested.zarr` for nested folders), or a zipped zarr file (file that ends with `.zarr.zip`)')
                aprint(f"Opening Zarr storage with mode='{mode}'")
                self._root_group = zarr.convenience.open(self._store, mode=mode)
                self._root_group.attrs['store_type'] = store_type
                # print(self._root_group.tree())
            except Exception as e:
                raise ValueError(f"Problem: can't create target file/directory, most likely the target dataset already exists or path incorrect: {path}")
        else:
            raise ValueError(f"Invalid read/write mode or invalid path: {path} (check path!)")

    def _initialise_existing(self, path: str):
        self._channels = [channel for channel, _ in self._root_group.groups()]

        # print(f"Exploring Zarr hierarchy...")
        for channel, channel_group in self._root_group.groups():
            # print(f"Found channel: {channel}")

            channel_items = channel_group.items()

            for item_name, array in channel_items:
                # print(f"Found array: {item_name}")

                if item_name == channel or item_name == 'fused':
                    # print(f'Opening array at {path}:{channel}/{item_name} ')
                    self._arrays[channel] = array
                    # self._arrays[channel] = from_zarr(path, component=f"{channel}/{item_name}")

    def _get_group_for_channel(self, channel: str) -> Union[None, Sequence[Group]]:
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

    def check_integrity(self, channels: Sequence[str] = None) -> bool:
        aprint(f"Checking integrity of zarr storage, might take some time.")
        if channels is None:
            channels = self.channels()
        for channel in channels:
            aprint(f"Checking integrity of channel '{channel}'...")
            array = self.get_array(channel, wrap_with_dask=False)
            if array.nchunks_initialized < array.nchunks:
                aprint(f"WARNING! not all chunks initialised! (dtype={array.dtype})")
                return False
            else:
                aprint(f"Channel '{channel}' seems ok!")
                return True

    def channels(self) -> Sequence[str]:
        return list(self._arrays.keys())

    def shape(self, channel: str) -> Sequence[int]:
        return self._arrays[channel].shape

    def chunks(self, channel: str) -> Sequence[int]:
        return self._arrays[channel].chunks

    def nb_timepoints(self, channel: str) -> int:
        return self.get_array(channel).shape[0]

    def dtype(self, channel: str):
        return self.get_array(channel).dtype

    def tree(self) -> str:
        return self._root_group.tree()

    def info(self, channel: str = None) -> str:
        if channel:
            info_str = f"Channel: '{channel}'', nb time points: {self.shape(channel)[0]}, shape: {self.shape(channel)[1:]}"
            info_str += "\n"
            info_str += str(self._arrays[channel].info)
            return info_str
        else:
            info_str = "Zarr tree: \n"
            info_str += str(self._root_group.tree())
            info_str += "\n\n"
            info_str += "Channels: \n"
            for channel in self.channels():
                info_str += "  └──" + self.info(channel) + "\n\n"

            return info_str

    def get_metadata(self):
        """get the attributes stored in the zarr folder"""
        attrs = {}
        for name in self._root_group.attrs:
            attrs[name] = self._root_group.attrs[name]
        return attrs

    def get_array(self, channel: str, per_z_slice: bool = False, wrap_with_dask: bool = False):
        array = self._arrays[channel]
        if wrap_with_dask:
            return dask.array.from_array(array, chunks=array.chunks)
        else:
            return array

    def get_stack(self, channel: str, time_point: int, per_z_slice: bool = False):
        stack_array = self.get_array(channel)[time_point]
        return stack_array

    def add_channel(self, name: str, shape: Tuple[int, ...], dtype, chunks: Tuple[int, ...] = None, codec: str = 'zstd', clevel: int = 3) -> Any:
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

        # check if channel exists:
        if name in self.channels():
            raise ValueError("Channel already exist!")

        if chunks is None:
            chunks = BaseDataset._default_chunks[0: len(shape)]

        aprint(f"chunks={chunks}")

        # Choosing the fill value to the largest value:
        fill_value = self._get_largest_dtype_value(dtype)

        aprint(f"Adding channel: '{name}' of shape: {shape}, chunks:{chunks}, dtype: {dtype}, fill_value: {fill_value}, codec: {codec}, clevel: {clevel} ")
        compressor = Blosc(cname=codec, clevel=clevel, shuffle=Blosc.BITSHUFFLE)
        filters = []

        channel_group = self._root_group.create_group(name)
        array = channel_group.full(name=name,
                                   shape=shape,
                                   dtype=dtype,
                                   chunks=chunks,
                                   filters=filters,
                                   compressor=compressor,
                                   fill_value=fill_value)

        self._arrays[name] = array

        return array

    def add_channels_to(self,
                        path: str,
                        channels: Sequence[str],
                        rename: Sequence[str],
                        store: str,
                        overwrite: bool
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

        zdataset = ZDataset(path, 'a', store)
        root = zdataset._root_group

        aprint(f"Existing channels: {zdataset.channels()}")

        for channel, new_name in zip(channels, rename):
            try:
                array = self.get_array(channel, per_z_slice=False)
                source_group = self._get_group_for_channel(channel)
                source_array = tuple((a for n, a in source_group.items() if n == channel))[0]

                aprint(f"Creating group for channel {channel} of new name {new_name}.")
                if new_name not in root.group_keys():
                    dest_group = root.create_group(new_name)
                else:
                    dest_group = root[new_name]

                aprint(f"Fast copying channel {channel} renamed to {new_name} of shape {array.shape} and dtype {array.dtype} ")
                convenience.copy(source_array, dest_group, if_exists='replace' if overwrite else 'raise')
            except CopyError | NotImplementedError:
                aprint(f"Channel already exists, set option '-w' to force overwriting! ")
