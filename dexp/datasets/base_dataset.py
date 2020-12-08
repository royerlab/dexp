from abc import ABC, abstractmethod
from typing import Sequence

import numpy


class BaseDataset(ABC):
    _default_chunks = (1, 128, 512, 512)

    def __init__(self, dask_backed=False):
        """ Instanciates a Base Dataset

        """
        self.dask_backed = dask_backed

    def _selected_channels(self, channels):
        if channels is None:
            selected_channels = self._channels
        else:
            selected_channels = list(set(channels) & set(self._channels))

        # aprint(f"Available channels: {self._channels}")
        # aprint(f"Requested channels: {channels if channels else '--All--'} ")
        # aprint(f"Selected channels:  {selected_channels}")

        return selected_channels

    def _get_largest_dtype_value(self, dtype):
        if numpy.issubdtype(dtype, numpy.integer):
            return numpy.iinfo(dtype).max
        elif numpy.issubdtype(dtype, numpy.float):
            return numpy.finfo(dtype).max

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def channels(self) -> Sequence[str]:
        pass

    @abstractmethod
    def shape(self, channel: str) -> Sequence[int]:
        pass

    @abstractmethod
    def chunks(self, channel: str) -> Sequence[int]:
        pass

    @abstractmethod
    def dtype(self, channel: str):
        pass

    @abstractmethod
    def tree(self) -> str:
        pass

    @abstractmethod
    def info(self, channel: str = None) -> str:
        pass

    @abstractmethod
    def get_metadata(self):
        pass

    @abstractmethod
    def get_array(self, channel: str, per_z_slice: bool = False, wrap_with_dask: bool = False):
        pass

    @abstractmethod
    def get_stack(self, channel: str, time_point: int, per_z_slice: bool = False):
        pass

    @abstractmethod
    def check_integrity(self, channels: Sequence[str]) -> bool:
        pass
