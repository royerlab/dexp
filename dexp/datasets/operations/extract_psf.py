from typing import Sequence, Union

import numpy as np
from arbol import aprint, asection

from dexp.datasets.base_dataset import BaseDataset
from dexp.processing.backends.best_backend import BestBackend
from dexp.utils.slicing import slice_from_shape
from dexp.processing.remove_beads import BeadsRemover


def dataset_extract_psf(dataset: BaseDataset,
                        dest_path: str,
                        channels: Sequence[str],
                        slicing: Union[Sequence[slice], slice],
                        peak_threshold: int = 500,
                        similarity_threshold: float = 0.5,
                        psf_size: int = 35,
                        device: int = 0,
                        stop_at_exception: bool = True,
                        verbose: bool = True,
                        ) -> None:
    """
    Computes PSF from beads.
    Additional information at dexp.processing.remove_beads.beadremover documentation.
    """
    dest_path = dest_path.split('.')[0]

    remove_beads = BeadsRemover(
        peak_threshold=peak_threshold,
        similarity_threshold=similarity_threshold,
        psf_size=psf_size,
        verbose=verbose
    )

    for channel in dataset._selected_channels(channels):
        array = dataset.get_array(channel)
        _, volume_slicing, time_points = slice_from_shape(array.shape, slicing)

        psfs = []
        for i in range(len(time_points)):
            tp = time_points[i]
            try:
                with asection(f'Removing beads of channel: {channel}'):
                    with asection(f'Loading time point {i}/{len(time_points)}'):
                        tp_array = np.asarray(array[tp][volume_slicing])

                    with asection('Processing'):
                        with BestBackend(exclusive=True, enable_unified_memory=True, device_id=device) as backend:
                            tp_array = backend.to_backend(tp_array)
                            estimated_psf = remove_beads.detect_beads(tp_array)

                aprint(f"Done extracting PSF from time point: {i}/{len(time_points)} .")

            except Exception as error:
                aprint(error)
                aprint(f"Error occurred while processing time point {i} !")
                import traceback
                traceback.print_exc()

                if stop_at_exception:
                    raise error
            
            psfs.append(estimated_psf)

        psfs = np.stack(psfs).mean(axis=0)
        print(psfs.shape)
        np.save(dest_path + channel + '.npy', psfs)
