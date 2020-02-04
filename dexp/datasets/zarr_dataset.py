from multiprocessing.pool import ThreadPool

import dask
import zarr
from numcodecs import blosc
from dexp.datasets.base_dataset import BaseDataset
import multiprocessing

nb_threads = 2*multiprocessing.cpu_count()
dask.config.set(scheduler='threads')
dask.config.set(pool=ThreadPool(nb_threads))
blosc.set_nthreads(nb_threads)

class ZDataset(BaseDataset):

    def __init__(self, folder, mode='r', cache_active=False, cache_size=int(8e9), use_dask=False):

        super().__init__(dask_backed=False)

        self._folder = folder

        print(f"Initialising Zarr storage...")
        store = zarr.storage.DirectoryStore(folder)

        if cache_active:
            print(f"Setting up Zarr cache...")
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

                if item_name == channel:
                    if use_dask:
                        self._arrays[channel] = dask.array.from_zarr(folder, component=f"{channel}/{item_name}")
                    else:
                        self._arrays[channel] = array

    def channels(self):
        return list(self._channels)

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



