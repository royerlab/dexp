import zarr

from dexp.datasets.dataset_base import DatasetBase


class ZDataset(DatasetBase):

    def __init__(self, folder, mode='r', cache_size=8e9):

        self._folder = folder

        store = zarr.storage.DirectoryStore(folder)
        cache_store = zarr.LRUStoreCache(store, max_size=cache_size)

        self._z = zarr.convenience.open(cache_store, mode=mode)
        self._channels = [channel for channel, _ in self._z.groups()]
        self._arrays = {}

        for channel, channel_group in self._z.groups():

            channel_items = channel_group.items()

            for item_name, array in channel_items:

                if item_name == channel:
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

    def get_stacks(self, channel):

        return self._arrays[channel]

    def get_stack(self, channel, time_point):

        stack_array = self.get_stacks(channel)[time_point]
        return stack_array

    def copy(self,
             path,
             channels=None,
             slice=None,
             compression='zstd',
             compression_level=3,
             chunk_size=64,
             overwrite=False,
             project=None):

        raise NotImplemented("This is not yet implemented!")
