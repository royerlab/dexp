from typing import Optional, List, Sequence

from arbol.arbol import aprint
from arbol.arbol import asection
from joblib import Parallel, delayed
from zarr.errors import ContainsArrayError, ContainsGroupError

from dexp.datasets.base_dataset import BaseDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.deskew.classic_deskew import classic_deskew
from dexp.processing.deskew.yang_deskew import yang_deskew


def dataset_deskew(dataset: BaseDataset,
                   dest_path: str,
                   channels: Sequence[str],
                   slicing,
                   dx: Optional[float] = None,
                   dz: Optional[float] = None,
                   angle: Optional[float] = None,
                   flips: Optional[Sequence[bool]] = None,
                   camera_orientation: int = 0,
                   depth_axis: int = 0,
                   lateral_axis: int = 1,
                   mode: str = 'classic',
                   padding: bool = True,
                   store: str = 'dir',
                   compression: str = 'zstd',
                   compression_level: int = 3,
                   overwrite: bool = False,
                   workers: int = 1,
                   workersbackend: str = 'threading',
                   devices: Optional[List[int]] = None,
                   check: bool = True,
                   stop_at_exception=True):

    # Collect arrays for selected channels:
    arrays = tuple(dataset.get_array(channel, per_z_slice=False, wrap_with_dask=True) for channel in channels)

    # Notify of selected channels and corresponding properties:
    with asection(f"Channels:"):
        for view, channel in zip(arrays, channels):
            aprint(f"Channel: {channel} of shape: {view.shape} and dtype: {view.dtype}")

    # Slicing:
    if slicing is not None:
        aprint(f"Slicing with: {slicing}")
        arrays = tuple(view[slicing] for view in arrays)

    # Default flipping:
    if flips is None:
        flips = (False,) * len(arrays)

    # Default devices:
    if devices is None:
        devices = [0]

    # We allocate last minute once we know the shape...
    from dexp.datasets.zarr_dataset import ZDataset
    zarr_mode = 'w' + ('' if overwrite else '-')
    dest_dataset = ZDataset(dest_path, zarr_mode, store, parent=dataset)

    # Metadata for deskewing:
    metadata = dataset.get_metadata()
    aprint(f"Dataset metadata: {metadata}")
    if dx is None and 'res' in metadata:
        dx = float(metadata['res'])
    if dz is None and 'dz' in metadata:
        dz = float(metadata['dz']) if dz is None else dz
    if angle is None and 'angle' in metadata:
        angle = float(metadata['angle']) if angle is None else angle

    aprint(f"Deskew parameters: dx={dx}, dz={dz}, angle={angle}")

    # Iterate through channels::
    for array, channel, flip in zip(arrays, channels, flips):

        # shape and dtype of views to deskew:
        shape = array[0].shape
        nb_timepoints = array.shape[0]

        def process(channel, tp, device):
            try:

                with asection(f"Loading channel {channel} for time point {tp}"):
                    array_tp = array[tp].compute()

                with BestBackend(device, exclusive=True, enable_unified_memory=True):

                    if 'yang' in mode:
                        deskewed_view_tp = yang_deskew(image=array_tp,
                                                        depth_axis=depth_axis,
                                                        lateral_axis=lateral_axis,
                                                        flip_depth_axis=flip,
                                                        dx=dx,
                                                        dz=dz,
                                                        angle=angle,
                                                        camera_orientation=camera_orientation)
                    elif 'classic' in mode:
                        deskewed_view_tp = classic_deskew(image=array_tp,
                                                           depth_axis=depth_axis,
                                                           lateral_axis=lateral_axis,
                                                           flip_depth_axis=flip,
                                                           dx=dx,
                                                           dz=dz,
                                                           angle=angle,
                                                           camera_orientation=camera_orientation,
                                                           padding=padding)
                    else:
                        raise ValueError(f"Deskew mode: {mode} not supported.")

                    with asection(f"Moving array from backend to numpy."):
                        deskewed_view_tp = Backend.to_numpy(deskewed_view_tp, dtype=array.dtype, force_copy=False)

                if channel not in dest_dataset.channels():
                    try:
                        dest_dataset.add_channel(channel,
                                                 shape=(array.shape[0],)+deskewed_view_tp.shape,
                                                 dtype=deskewed_view_tp.dtype,
                                                 codec=compression,
                                                 clevel=compression_level)
                    except (ContainsArrayError, ContainsGroupError):
                        aprint(f"Other thread/process created channel before... ")

                with asection(f"Saving fused stack for time point {tp}, shape:{deskewed_view_tp.shape}, dtype:{deskewed_view_tp.dtype}"):
                    dest_dataset.write_stack(channel=channel,
                                             time_point=tp,
                                             stack_array=deskewed_view_tp)

                aprint(f"Done processing time point: {tp} .")

            except Exception as error:
                aprint(error)
                aprint(f"Error occurred while deskewing time point {tp} !")
                import traceback
                traceback.print_exc()

                if stop_at_exception:
                    raise error

        if workers == -1:
            workers = len(devices)
        aprint(f"Number of workers: {workers}")

        if workers > 1:
            Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(channel, tp, devices[tp % len(devices)]) for tp in range(0, shape[0]))
        else:
            for tp in range(0, nb_timepoints):
                process(channel, tp, devices[0])

    # Dataset info:
    aprint(dest_dataset.info())

    # Check dataset integrity:
    if check:
        dest_dataset.check_integrity()

    # close destination dataset:
    dest_dataset.close()
