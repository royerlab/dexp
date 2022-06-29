from typing import Optional, Tuple

import numpy as np
import zarr

from dexp.utils.slicing import slice_from_shape


class StackIterator:
    def __init__(self, array: zarr.Array, slicing: Optional[slice]):

        self._out_shape, self._volume_slicing, self._time_points = slice_from_shape(array.shape, slicing)
        self._slicing = slicing

        self._array = array

    @property
    def shape(self) -> Tuple[int]:
        return self._out_shape

    @property
    def dtype(self) -> np.dtype:
        dtype = self._array.dtype
        if not isinstance(dtype, np.dtype):
            # converting tensorstore dtype
            dtype = dtype.numpy_dtype
        return dtype

    @property
    def slicing(self) -> Tuple[slice]:
        return self._slicing

    def __len__(self) -> int:
        return len(self._time_points)

    def __getitem__(self, index: int) -> np.ndarray:
        if isinstance(self._volume_slicing, type(...)):
            slicing = self._time_points[index]
        else:
            slicing = (self._time_points[index],) + self._volume_slicing
        return self._array[slicing]

    def time(self, index: int) -> int:
        return self._time_points[index]
