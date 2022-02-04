import cupy as cp
import numpy as np
from arbol import asection
from cucim.skimage import morphology

from dexp.datasets import ZDataset
from dexp.processing.morphology import area_white_top_hat


def remove_background(stack: np.ndarray) -> np.ndarray:
    with asection("Computing opening ..."):
        cu_stack = cp.asarray(stack)
        cu_opened = morphology.opening(cu_stack, morphology.ball(np.sqrt(2)))
        opened = cu_opened.get()

    with asection("Computing white top hat ..."):
        wth = area_white_top_hat(opened, area_threshold=10_000, sampling=4, axis=0)

    return wth


in_ds = ZDataset("demo_4views.zarr.zip")
out_ds = ZDataset("processed.zarr", mode="w-")

for channel in in_ds.channels():
    new_channel = channel + "-processed"
    out_ds.add_channel(name=new_channel, shape=in_ds.shape(channel), dtype=np.int32)
    with asection(f"Processing channel {channel} ..."):
        for tp in range(in_ds.nb_timepoints(channel)):
            stack = in_ds.get_stack(channel=channel, time_point=tp)
            labels = remove_background(stack)
            out_ds.write_stack(channel=new_channel, time_point=tp, stack_array=labels)

in_ds.close()
out_ds.close()
