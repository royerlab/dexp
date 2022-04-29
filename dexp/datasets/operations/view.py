from typing import List, Sequence, Union

import dask
import numpy as np
from arbol import aprint
from dask.array import reshape

from dexp.datasets import ZDataset
from dexp.datasets.base_dataset import BaseDataset


def dataset_view(
    input_dataset: BaseDataset,
    channels: Sequence[str],
    scale: int,
    contrast_limits: Union[List[int], List[float]],
    colormap: str,
    name: str,
    windowsize: int,
    projections_only: bool,
    volume_only: bool,
    quiet: bool,
):

    import napari
    import tensorstore as ts
    from napari.layers.utils._link_layers import (
        _get_common_evented_attributes,
        link_layers,
    )

    if quiet:
        from napari.components.viewer_model import ViewerModel

        viewer = ViewerModel()
    else:
        viewer = napari.Viewer(title=f"DEXP | viewing with napari: {name} ", ndisplay=2)
        viewer.window.resize(windowsize + 256, windowsize)

    viewer.grid.enabled = True

    scale_array = np.asarray((1, scale, scale, scale))

    for channel in channels:
        aprint(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")
        aprint(input_dataset.info(channel))

        resolution = np.asarray(input_dataset.get_resolution(channel))

        if isinstance(input_dataset, ZDataset):
            array = input_dataset.get_array(channel, wrap_with_tensorstore=True)
            if scale != 1:
                array = ts.downsample(array, scale_array, "mean")
                resolution *= scale_array
        else:
            array = input_dataset.get_array(channel, wrap_with_dask=True)

        layers = []
        if not projections_only:

            aprint(f"Adding array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

            # flip x for second camera:
            if "C1" in channel:
                if isinstance(array, ts.TensorStore):
                    array = array[..., ::-1]
                else:
                    array = dask.array.flip(array, -1)

            layer = viewer.add_image(
                array,
                name=channel,
                contrast_limits=contrast_limits,
                blending="additive",
                colormap=colormap,
                rendering="attenuated_mip",
                scale=resolution,
            )
            layers.append(layer)

        try:
            if not volume_only:
                for axis in (1, 2, 0):
                    if isinstance(array, ts.TensorStore):
                        proj_array = input_dataset.get_projection_array(channel, axis=axis, wrap_with_tensorstore=True)
                    else:
                        proj_array = input_dataset.get_projection_array(channel, axis=axis, wrap_with_dask=True)

                    # if the data format does not support projections we skip:
                    if proj_array is None:
                        continue

                    if isinstance(array, ts.TensorStore):
                        if axis == 1:
                            proj_array = proj_array[..., ::-1, :]  # flipping y

                        elif axis == 2:
                            # rotating 90 degrees on (2,1)-plane, equivalent to transposing (1,2) -> (-2, 1)
                            proj_array = proj_array[
                                ts.IndexTransform(
                                    input_rank=proj_array.ndim,
                                    output=[
                                        ts.OutputIndexMap(input_dimension=0),
                                        ts.OutputIndexMap(input_dimension=2, stride=-1),
                                        ts.OutputIndexMap(input_dimension=1),
                                    ],
                                )
                            ]

                        proj_array = proj_array[:, np.newaxis]  # appending dummy dim

                        if scale != 1:
                            proj_array = ts.downsample(proj_array, scale_array, "mean")

                    else:
                        if axis == 1:
                            proj_array = dask.array.flip(proj_array, 1)  # flipping y
                        elif axis == 2:
                            proj_array = dask.array.rot90(proj_array, axes=(2, 1))

                        shape = (
                            proj_array.shape[0],
                            1,
                        ) + proj_array.shape[1:]

                        proj_array = reshape(proj_array, shape=shape)

                    # flip x for second camera:
                    if "C1" in channel:
                        if isinstance(array, ts.TensorStore):
                            array = array[..., ::-1]
                        else:
                            array = dask.array.flip(array, -1)

                    proj_layer = viewer.add_image(
                        proj_array,
                        name=f"{channel}_proj_{axis}",
                        contrast_limits=contrast_limits,
                        blending="additive",
                        colormap=colormap,
                        scale=resolution,
                    )
                    layers.append(proj_layer)

        except KeyError:
            aprint("Warning: can't find projections!")

        if layers:
            attr = _get_common_evented_attributes(layers)
            attr.remove("visible")
            link_layers(layers, attr)

        viewer.dims.set_point(1, 0)  # moving to z = 0

    if not quiet:
        napari.run()
