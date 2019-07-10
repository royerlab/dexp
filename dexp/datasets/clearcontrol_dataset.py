import re
from fnmatch import fnmatch
from os import listdir, cpu_count
from os.path import join

import numcodecs
from cachey import Cache
from dask import array, delayed
from numcodecs.blosc import Blosc
from numpy import uint16, frombuffer
from zarr import open_group

from dexp.datasets.dataset_base import DatasetBase

numcodecs.blosc.use_threads = True
numcodecs.blosc.set_nthreads(cpu_count() // 2)


class CCDataset(DatasetBase):

    def __init__(self, folder, cache_size=8e9):

        self.folder = folder
        self._channels = []
        self._index_files = {}

        all_files = list(listdir(folder))
        print(all_files)

        for file in all_files:
            if fnmatch(file, '*.index.txt'):
                if not file.startswith('._'):
                    channel = file.replace('.index.txt', '')
                    self._channels.append(channel)
                    self._index_files[channel] = join(folder, file)

        print(self._channels)
        print(self._index_files)

        self._nb_time_points = {}
        self._times_sec = {}
        self._shapes = {}
        self._time_points = {}

        for channel in self._channels:
            self._parse_channel(channel)

        self.cache = Cache(cache_size)  # Leverage two gigabytes of memory

    def channels(self):
        return list(self._channels)

    def shape(self, channel, time_point=0):
        if (channel, time_point) in self._shapes:
            return self._shapes[(channel, time_point)]
        else:
            return ()

    def nb_timepoints(self, channel):
        return self._nb_time_points[channel]

    def get_stacks(self, channel):

        # Lazy version of get_stack:
        lazy_get_stack = delayed(self.cache.memoize(self.get_stack), pure=True)

        # Lazily load each stack for each time point:
        lazy_stacks = [lazy_get_stack(channel, time_point) for time_point in self._time_points[channel]]

        dtype = uint16
        shape = self._shapes[(channel, 0)]

        # Construct a small Dask array for every lazy value:
        arrays = [array.from_delayed(lazy_stack,
                                     dtype=dtype,
                                     shape=shape)
                  for lazy_stack in lazy_stacks]

        whole_array = array.stack(arrays, axis=0)  # Stack all small Dask arrays into one

        return whole_array

    def get_stack(self, channel, time_point):

        file_name = self._get_stack_file_name(channel, time_point)
        shape = self._shapes[(channel, time_point)]
        stack = self._get_array_for_stack_file(file_name, shape=shape)

        return stack

    def to_zarr(self, path, channels=None, slice=None, compression='zstd', compression_level=3, chunk_size=64, overwrite=False):

        mode = 'w' + ('' if overwrite else '-')
        root = open_group(path, mode=mode)

        filters = []  # [Delta(dtype='i4')]
        compressor = Blosc(cname=compression, clevel=compression_level, shuffle=Blosc.BITSHUFFLE)

        if channels is None:
            selected_channels = self._channels
        else:
            selected_channels = list(set(channels) & set(self._channels))

        print(f"Available channels: {self._channels}")
        print(f"Requested channels: {channels}")
        print(f"Selected channels:  {selected_channels}")

        for channel in selected_channels:

            channel_group = root.create_group(channel)

            array = self.get_stacks(channel)

            if not slice is None:
                array = array[slice]

            dim = len(array.shape)

            if dim <= 3:
                chunks = (chunk_size,) * dim
            else:
                chunks = (1,) * (dim - 3) + (chunk_size,) * 3

            print(f"Writing Zarr array for channel '{channel}' of shape {array.shape} ")

            z = channel_group.array(channel,
                                    data=array,
                                    chunks=chunks,
                                    filters=filters,
                                    compressor=compressor)

        # print(root.info)
        print("Zarr tree:")
        print(root.tree())

    def _parse_channel(self, channel):

        index_file = self._index_files[channel]

        with open(index_file) as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]

        lines = [re.split(r'\t+', line) for line in lines]

        self._time_points[channel] = []

        for line in lines:
            time_point = int(line[0])
            time_sec = float(line[1])
            shape = eval('(' + line[2] + ')')[::-1]

            self._times_sec[(channel, time_point)] = time_sec
            self._shapes[(channel, time_point)] = shape

            self._time_points[channel].append(time_point)

        self._nb_time_points[channel] = len(self._time_points[channel])

    def _get_stack_file_name(self, channel, time_point):

        return join(self.folder, 'stacks', channel, str(time_point).zfill(6) + '.raw')

    def _get_array_for_stack_file(self, file_name, shape=None):

        with open(file_name, 'rb') as my_file:
            print(f"Accessing file: {file_name}")
            buffer = my_file.read()
            array = frombuffer(buffer, dtype=uint16)

        if not shape is None:
            array = array.reshape(shape)

        return array
