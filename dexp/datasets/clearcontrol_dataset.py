import os
import re
from fnmatch import fnmatch
from os import listdir
from os.path import exists, join
from typing import Any, List, Sequence, Tuple

import numpy as np
from arbol.arbol import aprint
from cachey import Cache
from dask import array, delayed

from dexp.datasets.base_dataset import BaseDataset
from dexp.io.compress_array import decompress_array
from dexp.utils.config import config_blosc


class CCDataset(BaseDataset):
    def __init__(self, path, cache_size=8e9):

        super().__init__(dask_backed=False, path=path)

        config_blosc()

        self._channels = []
        self._index_files = {}

        all_files = list(listdir(path))
        # print(all_files)

        for file in all_files:
            if fnmatch(file, "*.index.txt"):
                if not file.startswith("._"):
                    channel = file.replace(".index.txt", "")
                    self._channels.append(channel)
                    self._index_files[channel] = join(path, file)

        # print(self._channels)
        # print(self._index_files)

        self._nb_time_points = {}
        self._times_sec = {}
        self._shapes = {}
        self._channel_shape = {}
        self._time_points = {}

        self._is_compressed = self._find_out_if_compressed(path)

        for channel in self._channels:
            self._parse_channel(channel)

        self.cache = Cache(cache_size)  # Leverage two gigabytes of memory

    def _parse_channel(self, channel):

        index_file = self._index_files[channel]

        with open(index_file) as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]

        lines = [re.split(r"\t+", line) for line in lines]

        self._time_points[channel] = []
        self._times_sec[channel] = []
        self._shapes[channel] = []

        for line in lines:
            time_point = int(line[0])
            time_sec = float(line[1])
            shape = eval("(" + line[2] + ")")[::-1]

            self._times_sec[channel].append(time_sec)
            self._shapes[channel].append(shape)

            if channel in self._channel_shape:
                existing_shape = self._channel_shape[channel]
                if shape != existing_shape:
                    aprint(
                        f"Warning: Channel {channel} has varying stack shape! Shape changes from "
                        + f"{existing_shape} to {shape} at time point {time_point}"
                    )
            self._channel_shape[channel] = shape

            self._time_points[channel].append(time_point)

        self._nb_time_points[channel] = len(self._time_points[channel])

    def _get_stack_file_name(self, channel, time_point):

        compressed_file_name = join(self._path, "stacks", channel, str(time_point).zfill(6) + ".blc")
        raw_file_name = join(self._path, "stacks", channel, str(time_point).zfill(6) + ".raw")

        if self._is_compressed is None:
            if exists(compressed_file_name):
                self._is_compressed = True
            else:
                self._is_compressed = False

        if self._is_compressed:
            return compressed_file_name
        else:
            return raw_file_name

    def _get_array_for_stack_file(self, file_name, shape=None, dtype=None):

        try:
            if file_name.endswith(".raw"):
                aprint(f"Accessing file: {file_name}")

                dt = np.dtype(np.uint16)
                dt = dt.newbyteorder("L")

                array = np.fromfile(file_name, dtype=dt)

            elif file_name.endswith(".blc"):
                array = np.empty(shape=shape, dtype=dtype)
                with open(file_name, "rb") as binary_file:
                    # Read the whole file at once
                    data = binary_file.read()
                    decompress_array(data, array)

            # Reshape array:
            if shape is not None:
                array = array.reshape(shape)

            return array

        except FileNotFoundError:
            aprint(f"Could not find file: {file_name} for array of shape: {shape}")
            return np.zeros(shape, dtype=np.uint16)

    def _get_slice_array_for_stack_file_and_z(self, file_name, shape, z):

        try:
            if file_name.endswith(".raw"):
                aprint(f"Accessing file: {file_name} at z={z}")

                length = shape[1] * shape[2] * np.dtype(np.uint16).itemsize
                offset = z * length

                dt = np.dtype(np.uint16)
                dt = dt.newbyteorder("L")

                array = np.fromfile(file_name, offset=offset, count=length, dtype=dt)
            elif file_name.endswith(".blc"):
                raise NotImplementedError("This type of access is not yet supported")

            array = array.reshape(shape[1:])
            return array

        except FileNotFoundError:
            aprint(f"Could  not find file: {file_name} for array of shape: {shape} at z={z}")
            return np.zeros(shape[1:], dtype=np.uint16)

    def close(self):
        # Nothing to do...
        pass

    def channels(self) -> List[str]:
        return list(self._channels)

    def shape(self, channel: str, time_point: int = 0) -> Sequence[int]:
        try:
            return (self._nb_time_points[channel],) + self._shapes[channel][time_point]
        except (IndexError, KeyError):
            return ()

    def dtype(self, channel: str):
        return np.uint16

    def info(self, channel: str = None) -> str:
        if channel:
            info_str = (
                f"Channel: '{channel}', nb time points: {self.shape(channel)[0]}, shape: {self.shape(channel)[1:]} "
            )
            return info_str
        else:
            return self.tree()

    def get_metadata(self):
        # TODO: implement this!
        return {}

    def append_metadata(self, metadata: dict):
        raise NotImplementedError("Method append_metadata is not available for a joined dataset!")

    def get_array(self, channel: str, per_z_slice: bool = True, wrap_with_dask: bool = False):

        # Lazy and memorized version of get_stack:
        lazy_get_stack = delayed(self.get_stack, pure=True)

        # Lazily load each stack for each time point:
        lazy_stacks = [lazy_get_stack(channel, time_point, per_z_slice) for time_point in self._time_points[channel]]

        # Construct a small Dask array for every lazy value:
        arrays = [
            array.from_delayed(lazy_stack, dtype=np.uint16, shape=self._channel_shape[channel])
            for lazy_stack in lazy_stacks
        ]

        stacked_array = array.stack(arrays, axis=0)  # Stack all small Dask arrays into one

        return stacked_array

    def get_stack(self, channel, time_point, per_z_slice=True, wrap_with_dask: bool = False):

        file_name = self._get_stack_file_name(channel, time_point)
        shape = self._shapes[channel][time_point]

        if per_z_slice and not self._is_compressed:

            lazy_get_slice_array_for_stack_file_and_z = delayed(
                self.cache.memoize(self._get_slice_array_for_stack_file_and_z), pure=True
            )

            # Lazily load each stack for each time point:
            lazy_stacks = [lazy_get_slice_array_for_stack_file_and_z(file_name, shape, z) for z in range(0, shape[0])]

            arrays = [array.from_delayed(lazy_stack, dtype=np.uint16, shape=shape[1:]) for lazy_stack in lazy_stacks]

            stack = array.stack(arrays, axis=0)

        else:
            stack = self._get_array_for_stack_file(file_name, shape=shape, dtype=np.uint16)

        return stack

    def add_channel(self, name: str, shape: Tuple[int, ...], dtype, enable_projections: bool = True, **kwargs) -> Any:
        raise NotImplementedError("Not implemented!")

    def get_projection_array(self, channel: str, axis: int, wrap_with_dask: bool = True) -> Any:
        return None

    def write_array(self, channel: str, array: np.ndarray):
        raise NotImplementedError("Not implemented!")

    def write_stack(self, channel: str, time_point: int, stack: np.ndarray):
        raise NotImplementedError("Not implemented!")

    def check_integrity(self, channels: Sequence[str]) -> bool:
        # TODO: actually implement!
        return True

    def _find_out_if_compressed(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".blc"):
                    return True

        return False

    def time_sec(self, channel: str) -> np.ndarray:
        return np.asarray(self._times_sec[channel])
