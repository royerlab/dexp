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
                 projections_only: bool,
                 volume_only: bool,
                 rescale_time: bool):

    import napari
    from napari.layers.utils._link_layers import link_layers, _get_common_evented_attributes

    if rescale_time:
        nb_time_points = [input_dataset.shape(channel)[0] for channel in input_dataset.channels()]
        max_time_points = max(nb_time_points)
        time_scales = [round(max_time_points / n) for n in nb_time_points]
    else:
        time_scales = [1] * len(input_dataset.channels())

    viewer = napari.Viewer(title=f"DEXP | viewing with napari: {name} ", ndisplay=2)
    viewer.grid.enabled = True
    viewer.window.resize(windowsize + 256, windowsize)

    for time_scale, channel in zip(time_scales, channels):
        aprint(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")
        aprint(input_dataset.info(channel))

        array = input_dataset.get_array(channel, wrap_with_dask=True)

        layers = []
        try:
            if not volume_only:
                for axis in range(array.ndim - 1):
                    proj_array = input_dataset.get_projection_array(channel, axis=axis, wrap_with_dask=True)

                    # if the data format does not support projections we skip:
                    if proj_array is None:
                        continue

                    if axis == 1:
                        proj_array = dask.array.flip(proj_array, 1)  # flipping y
                    elif axis == 2:
                        proj_array = dask.array.rot90(proj_array, axes=(2, 1))  #

                    shape = (proj_array.shape[0], 1,) + proj_array.shape[1:]
                    proj_array = reshape(proj_array, shape=shape)

                    if proj_array is not None:
                        proj_layer = viewer.add_image(proj_array,
                                                        name=channel + '_proj_' + str(axis),
                                                        contrast_limits=contrast_limits,
                                                        blending='additive',
                                                        colormap=colormap, )
                        layers.append(proj_layer)

                        if aspect is not None:
                            if axis == 0:
                                proj_layer.scale = (time_scale, 1, 1, 1)
                            elif axis == 1:
                                proj_layer.scale = (time_scale, 1, aspect, 1)
                            elif axis == 2:
                                proj_layer.scale = (time_scale, 1, 1, aspect)  # reordered due to rotation

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
            layers.append(layer)

            if aspect is not None:
                if array.ndim == 3:
                    layer.scale = (aspect, 1, 1)
                elif array.ndim == 4:
                    layer.scale = (time_scale, aspect, 1, 1)
                    aprint(f'Setting time scale to {time_scale}')
                aprint(f"Setting aspect ratio to {aspect} (layer.scale={layer.scale})")

        if layers:
            attr = _get_common_evented_attributes(layers)
            attr.remove('visible')
            link_layers(layers, attr)

    napari.run()
