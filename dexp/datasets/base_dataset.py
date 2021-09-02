from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Any

import numpy


class BaseDataset(ABC):

    def __init__(self, dask_backed=False):
        """ Instanciates a Base Dataset

        """
        self.dask_backed = dask_backed

    def _selected_channels(self, channels: Sequence[str]):
        if channels is None:
            selected_channels = self.channels()
        else:
            selected_channels = [channel for channel in channels if channel in self.channels()]

        # aprint(f"Available channels: {self._channels}")
        # aprint(f"Requested channels: {channels if channels else '--All--'} ")
        # aprint(f"Selected channels:  {selected_channels}")

        return selected_channels

    def _get_largest_dtype_value(self, dtype):
        if numpy.issubdtype(dtype, numpy.integer):
            return numpy.iinfo(dtype).max
        elif numpy.issubdtype(dtype, numpy.floating):
            return numpy.finfo(dtype).max
        else:
            raise NotImplementedError(f'Found {type(dtype), dtype}.')

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def channels(self) -> Sequence[str]:
        pass

    def nb_timepoints(self, channel: str) -> int:
        return self.shape(channel)[0]

    @abstractmethod
    def shape(self, channel: str) -> Sequence[int]:
        pass

    @abstractmethod
    def dtype(self, channel: str):
        pass

    def tree(self) -> str:
        tree_str = f"{type(self).__name__} dataset"
        tree_str += "\n\n"
        tree_str += "Channels: \n"
        for channel in self.channels():
            tree_str += "  └──" + self.info(channel) + "\n"
        return tree_str

    @abstractmethod
    def info(self, channel: str = None) -> str:
        pass

    @abstractmethod
    def get_metadata(self):
        pass

    @abstractmethod
    def append_metadata(self, metadata: dict):
        pass

    @abstractmethod
    def add_channel(self, name: str, shape: Tuple[int, ...], dtype, enable_projections: bool = True, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_array(self, channel: str, per_z_slice: bool = False, wrap_with_dask: bool = False):
        pass

    @abstractmethod
    def get_stack(self, channel: str, time_point: int, per_z_slice: bool = False, wrap_with_dask: bool = False):
        pass

    @abstractmethod
    def get_projection_array(self, channel: str, axis: int, wrap_with_dask: bool = False) -> Any:
        pass

    @abstractmethod
    def write_array(self, channel: str, array: numpy.ndarray):
        pass

    @abstractmethod
    def write_stack(self, channel: str, time_point: int, stack: numpy.ndarray):
        pass

    @abstractmethod
    def check_integrity(self, channels: Sequence[str]) -> bool:
        pass
