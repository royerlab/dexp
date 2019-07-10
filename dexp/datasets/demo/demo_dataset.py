from time import time

from napari import Viewer
from napari.util import app_context
from numpy import s_

from dexp.datasets.clearcontrol_dataset import CCDataset
from dexp.datasets.zarr_dataset import ZDataset


def demo(path):
    dataset = CCDataset(path)

    assert not dataset is None

    print(dataset.channels())

    print(dataset.nb_timepoints(dataset._channels[0]))

    time_start = time()
    first_stack_no_dask = dataset.get_stack('sequential', 0)
    time_stop = time()

    print(f"Elapsed time to load one stack: {time_stop - time_start} seconds")
    print(first_stack_no_dask.shape)

    array = dataset.get_stacks('sequential')
    print(array.shape)

    first_stack = array[0]
    print(first_stack.shape)

    time_start = time()
    dataset.to_zarr("/Users/royer/Downloads/testzarr.zarr", slice=s_[0:5], overwrite=True)
    time_stop = time()
    print(f"Elapsed time to save to Zarr: {time_stop - time_start} seconds")

    dataset = ZDataset("/Users/royer/Downloads/testzarr.zarr")

    print(dataset.channels())
    print(dataset.info('sequential'))
    print(dataset.get_stacks('sequential').shape)

    with app_context():
        viewer = Viewer()

        viewer.add_image(dataset.get_stacks('sequential'), name='image', clim_range=[0, 1000])


demo('/Users/royer/Volumes/micro1/data/prod/2019-07-03-16-23-15-65-f2va.mch7')
