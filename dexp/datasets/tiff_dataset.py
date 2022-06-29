from typing import Dict, Sequence, Tuple

import dask.array as da
import numpy as np
from dask.array.image import imread
from numpy.typing import ArrayLike

from dexp.datasets.base_dataset import BaseDataset


class TIFDataset(BaseDataset):
    def __init__(self, path: str):
        super().__init__(dask_backed=False, path=path)
        self._channel = "stack"
        self._array = imread(path)

    def _assert_channel(self, channel: str) -> None:
        if channel != self._channel:
            raise KeyError("TIFDataset only contains a single channel named `stack`.")

    def channels(self) -> Sequence[str]:
        return ["stack"]

    def close(self) -> None:
        pass

    def shape(self, channel: str) -> Sequence[int]:
        self._assert_channel(channel)
        return self._array.shape

    def dtype(self, channel: str) -> np.dtype:
        self._assert_channel(channel)
        return self._array.dtype

    def get_metadata(self) -> Dict:
        return {}

    def append_metadata(self, metadata: Dict) -> None:
        raise NotImplementedError

    def add_channel(self, name: str, shape: Tuple[int, ...], dtype, enable_projections: bool = True, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def _maybe_compute(array: da.Array, wrap_with_dask: bool) -> ArrayLike:
        return array if wrap_with_dask else array.compute()

    def get_array(self, channel: str, per_z_slice: bool = False, wrap_with_dask: bool = True) -> ArrayLike:
        self._assert_channel(channel)
        return self._maybe_compute(self._array, wrap_with_dask)

    def get_stack(
        self, channel: str, time_point: int, per_z_slice: bool = False, wrap_with_dask: bool = True
    ) -> ArrayLike:
        if per_z_slice:
            raise NotImplementedError

        self._assert_channel(channel)
        return self._maybe_compute(self._array[time_point], wrap_with_dask)

    def get_projection_array(self, channel: str, axis: int, wrap_with_dask: bool = True) -> ArrayLike:
        self._assert_channel(channel)
        return self._maybe_compute(self._array.max(axis=axis), wrap_with_dask)

    def write_array(self, channel: str, array: ArrayLike) -> None:
        raise NotImplementedError

    def write_stack(self, channel: str, time_point: int, stack: ArrayLike) -> None:
        raise NotImplementedError

    def check_integrity(self, channels: Sequence[str]) -> bool:
        return True
