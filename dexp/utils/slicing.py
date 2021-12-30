from typing import List, Tuple

import numpy as np


def slice_from_shape(shape: Tuple[int], slicing: slice) -> Tuple[Tuple[int], slice, List[int]]:
    time_points = list(range(shape[0]))
    if slicing is not None:
        if isinstance(slicing, tuple):
            time_points = time_points[slicing[0]]
            volume_slicing = slicing[1:]
            new_shape = (len(time_points),) + np.empty(shape=shape[1:], dtype=bool)[slicing[1:]].shape
        else:
            time_points = time_points[slicing]
            volume_slicing = ...
            new_shape = (len(time_points),) + shape[1:]
    else:
        volume_slicing = ...
        new_shape = shape
    return new_shape, volume_slicing, time_points
