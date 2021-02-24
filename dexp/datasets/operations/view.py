from typing import Sequence, Union, List

import dask
from arbol import aprint
from dask.array import reshape

from dexp.datasets.base_dataset import BaseDataset


def dataset_view(input_dataset: BaseDataset,
                 channels: Sequence[str],
                 slicing,
                 aspect: float,
                 contrast_limits: Union[List[int], List[float]],
                 colormap: str,
                 name: str,
                 windowsize: int,
                 projections_only):
    # Annoying napari induced warnings:
    import warnings
    warnings.filterwarnings("ignore")

    from napari import gui_qt, Viewer
    from napari._qt.qthreading import thread_worker
    with gui_qt():
        viewer = Viewer(title=f"DEXP | viewing with napari: {name} ", ndisplay=2)
        viewer.grid.enabled = True
        viewer.window.resize(windowsize + 256, windowsize)

        for channel in channels:
            aprint(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")
            aprint(input_dataset.info(channel))

            array = input_dataset.get_array(channel, wrap_with_dask=True)

            try:
                for axis in range(array.ndim - 1):
                    proj_array = input_dataset.get_projection_array(channel, axis=axis, wrap_with_dask=True)

                    shape = (proj_array.shape[0], 1,) + proj_array.shape[1:]
                    proj_array = reshape(proj_array, shape=shape)

                    if proj_array is not None:
                        proj_layer = viewer.add_image(proj_array,
                                                      name=channel + '_proj_' + str(axis),
                                                      contrast_limits=contrast_limits,
                                                      blending='additive',
                                                      colormap=colormap, )

                        if aspect is not None:
                            if axis == 0:
                                proj_layer.scale = (1, 1)
                            elif axis == 1:
                                proj_layer.scale = (aspect, 1)
                            elif axis == 2:
                                proj_layer.scale = (aspect, 1)

                            aprint(f"Setting aspect ratio for projection (layer.scale={proj_layer.scale})")

            except KeyError:
                aprint("Warning: can't find projections!")

            if not projections_only:

                if slicing:
                    array = array[slicing]

                aprint(f"Adding array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

                # flip x for second camera:
                if 'C1' in channel:
                    array = dask.array.flip(array, -1)

                layer = viewer.add_image(array,
                                         name=channel,
                                         contrast_limits=contrast_limits,
                                         blending='additive',
                                         colormap=colormap,
                                         attenuation=0.04,
                                         rendering='attenuated_mip')

                if aspect is not None:
                    if array.ndim == 3:
                        layer.scale = (aspect, 1, 1)
                    elif array.ndim == 4:
                        layer.scale = (1, aspect, 1, 1)
                    aprint(f"Setting aspect ratio to {aspect} (layer.scale={layer.scale})")

                # For some reason some parameters refuse to be set, this solves it:
                @thread_worker
                def workaround_for_recalcitrant_parameters():
                    aprint("Setting 3D rendering parameters")
                    layer.attenuation = 0.02
                    layer.rendering = 'attenuated_mip'

                worker = workaround_for_recalcitrant_parameters()
                worker.start()
