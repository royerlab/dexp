import zarr

from dexp.datasets.dataset_base import DatasetBase


class ZDataset(DatasetBase):

    def __init__(self, folder, mode='r'):

        self._folder = folder
        self._z = zarr.convenience.open(folder, mode=mode)
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

    def info(self, channel):
        return self._arrays[channel].info

    def get_stacks(self, channel):

        return self._arrays[channel]

    def get_stack(self, channel, time_point):

        stack_array = self.get_stacks(channel)[time_point]
        return stack_array
