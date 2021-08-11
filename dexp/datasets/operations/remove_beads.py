from typing import Sequence, Optional, List, Tuple, Union

import numpy
import numpy as np
from arbol import aprint, asection

import dask
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from dexp.datasets.base_dataset import BaseDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend  # FIXME remove this
from dexp.processing.backends.best_backend import BestBackend
from dexp.utils.slicing import slice_from_shape
from dexp.utils import xpArray

import scipy
import scipy.ndimage


def get_slice(coord: Tuple[int], size: int) -> List[slice]:
    return [slice(c - size // 2, c + size // 2 + 1) for c in coord]


def remove_beads(array: xpArray, quantile: float, size: int = 21,
                 sim_thold: float = 0.5) -> xpArray:
    # sp = Backend.current().get_sp_module()
    # xp = Backend.current().get_xp_module()
    xp = numpy
    sp = scipy

    kernel = xp.zeros((1, 5, 5))
    kernel[0, 2, 2] = 4
    kernel[0, 2, 0] = -1
    kernel[0, 0, 2] = -1
    kernel[0, 2, 4] = -1
    kernel[0, 4, 2] = -1

    array = xp.pad(array, pad_width=size // 2, mode='edge')

    dots = sp.ndimage.convolve(array.astype(xp.float32), weights=kernel)
    max_value = dots.max()
    print('max_value', max_value)
    coordinates = np.where(dots > max_value * (1.0 - quantile))
    coordinates = np.vstack(coordinates).T.copy()
    print(coordinates.shape)

    beads = [array[get_slice(coord, size)].reshape(-1) for coord in coordinates]

    avg_bead = xp.stack(beads).mean(axis=0)

    beads = [bead / xp.linalg.norm(bead) for bead in beads]
    avg_bead = avg_bead / xp.linalg.norm(avg_bead)

    import napari
    shape = (size,) * array.ndim
    viewer = napari.Viewer()
    viewer.add_image(avg_bead.reshape(shape))

    for bead in beads:
        angle = xp.dot(bead, avg_bead)
        viewer.add_image(bead.reshape(shape), name=f'Angle {angle}')
        if angle > sim_thold:
            pass
        print(f'Angle {angle}')

    napari.run()


def dataset_remove_beads(dataset: BaseDataset,
                         dest_path: str,
                         channels: Sequence[str],
                         slicing: Union[Sequence[slice], slice],
                         store: str = 'dir',
                         compression: str = 'zstd',
                         compression_level: int = 3,
                         overwrite: bool = False,
                         quantile: float = 0.05,
                         devices: Optional[List[int]] = None,
                         check: bool = True,
                         stop_at_exception: bool = True,
                         ) -> None:

    from dexp.datasets.zarr_dataset import ZDataset
    mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(dest_path, mode, store, parent=dataset)

    # CUDA DASK cluster
    # cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=devices)
    # client = Client(cluster)
    # aprint('Dask Client', client)

    lazy_computation = []

    for channel in dataset._selected_channels(channels):

        array = dataset.get_array(channel)
        out_shape, volume_slicing, time_points = slice_from_shape(array.shape, slicing)

        dest_array = dest_dataset.add_channel(name=channel,
                                              shape=out_shape,
                                              dtype=array.dtype,
                                              codec=compression,
                                              clevel=compression_level)

        # @dask.delayed
        def process(i):
            tp = time_points[i]

            try:
                with asection(f'Removing beads of channel: {channel}'):
                    with asection(f'Loading time point {i}/{len(time_points)}'):
                        tp_array = numpy.asarray(array[tp][volume_slicing])

                    with asection('Processing'):
                        # with BestBackend(exclusive=True, enable_unified_memory=True):
                        with NumpyBackend():  # FIXME
                            tp_array = remove_beads(tp_array, quantile)

                    with asection(f"Moving array from backend to numpy."):
                        tp_array = Backend.to_numpy(tp_array, dtype=dest_array.dtype, force_copy=False)

                with asection(f"Saving deconvolved stack for time point {i}, shape:{tp_array.shape}, dtype:{array.dtype}"):
                    dest_dataset.write_stack(channel=channel,
                                             time_point=i,
                                             stack_array=tp_array)

                aprint(f"Done processing time point: {i}/{len(time_points)} .")

            except Exception as error:
                aprint(error)
                aprint(f"Error occurred while processing time point {i} !")
                import traceback
                traceback.print_exc()

                if stop_at_exception:
                    raise error

        for i in range(len(time_points)):
            lazy_computation.append(process(i))

    # dask.compute(*lazy_computation)

    # Dataset info:
    aprint(dest_dataset.info())

    # Check dataset integrity:
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
    client.close()


if __name__ == '__main__':
    from dexp.datasets.zarr_dataset import ZDataset
    ds_path = '/mnt/micro1_nfs/_dorado/2021/June/PhotoM_06222021/ch0.fused.zarr'
    ds = ZDataset(ds_path)
    dataset_remove_beads(ds, 'deleteme.zarr', overwrite=True, channels=ds.channels(),
                         slicing=(slice(0, 1), slice(None, None), slice(1000, None), slice(1000, None)))
