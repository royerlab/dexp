from multiprocessing.pool import ThreadPool
from os.path import isfile, isdir

import dask
import psutil
import zarr
from numcodecs import blosc
from zarr import ZipStore, DirectoryStore, open_group, convenience

from dexp.datasets.base_dataset import BaseDataset
import multiprocessing

nb_threads = multiprocessing.cpu_count()//2
dask.config.set(scheduler='threads')
dask.config.set(pool=ThreadPool(nb_threads))
blosc.set_nthreads(nb_threads//2 - 1)

class ZDataset(BaseDataset):

    def __init__(self, path, mode='r', cache_active=False, cache_size=psutil.virtual_memory().available, use_dask=False):

        super().__init__(dask_backed=False)

        self._folder = path

        print(f"Initialising Zarr storage...")
        if isfile(path) and path.endswith('.zarr.zip'):
            store = zarr.storage.ZipStore(path)
        elif isdir(path) and ( path.endswith('.zarr') or path.endswith('.zarr/')):
            store = zarr.storage.DirectoryStore(path)
        else:
            print(f'Cannot open {path}, needs to be a zarr directory (directory that ends with `.zarr`), or a zipped zarr file (file that ends with `.zarr.zip`)')

        if cache_active:
            print(f"Setting up Zarr cache with {cache_size / 1e9} GB...")
            store = zarr.LRUStoreCache(store, max_size=cache_size)

        print(f"Opening Zarr storage...")
        self._z = zarr.convenience.open(store, mode=mode)
        self._channels = [channel for channel, _ in self._z.groups()]
        self._arrays = {}

        print(f"Exploring Zarr hierarchy...")
        for channel, channel_group in self._z.groups():
            print(f"Found channel: {channel}")

            channel_items = channel_group.items()

            for item_name, array in channel_items:
                print(f"Found array: {item_name}")

                if item_name == channel or item_name == 'fused':
                    if use_dask:
                        self._arrays[channel] = dask.array.from_zarr(path, component=f"{channel}/{item_name}")
                    else:
                        self._arrays[channel] = array

    def channels(self):
        return list(self._channels)

    def get_group_for_channel(self, channel):
        groups = [g for c, g in self._z.groups() if c == channel]
        if len(groups)==0:
            return None
        else:
            return groups[0]

    def shape(self, channel):
        return self._arrays[channel].shape

    def nb_timepoints(self, channel):
        return self.get_stacks(channel).shape[0]

    def info(self, channel=None):

        if channel:
            info_str = f"Channel: '{channel}'', nb time points: {self.nb_timepoints(channel)}, shape: {self.shape(channel)}"
            info_str += "\n"
            info_str += str(self._arrays[channel].info)
            return info_str
        else:
            info_str = str(self._z.tree())
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



    def add(self,
            path,
            channels,
            rename,
            store,
            overwrite
            ):

        try:
            print(f"opening Zarr file for writing at: {path}")
            if store == 'zip':
                path = path if path.endswith('.zip') else path+'.zip'
                store = ZipStore(path)
            elif  store == 'dir':
                store = DirectoryStore(path)
            root = open_group(store, mode='a')
        except Exception as e:
            print(
                f"Problem: can't create target file/directory, most likely the target dataset already exists or path incorrect.")
            return None

        for channel, new_name in zip(channels, rename):

            array = self.get_stacks(channel, per_z_slice=False)
            source_group = self.get_group_for_channel(channel)
            source_array = tuple((a for n,a in source_group.items() if n == channel))[0]

            print(f"Creating group for channel {channel} of new name {new_name}.")
            dest_group = root.create_group(new_name)

            print(f"Fast copying channel {channel} renamed to {new_name} of shape {array.shape} and dtype {array.dtype} ")
            convenience.copy(source_array, dest_group, if_exists='replace' if overwrite else 'raise')



