from typing import Sequence, Tuple

from arbol import aprint, asection
from zarr.errors import ArrayNotFoundError


def dataset_view_remote(input_path: str,
                        name: str,
                        aspect: float,
                        channels: Sequence[str],
                        contrast_limits: Tuple[float, float],
                        colormap: str,
                        slicing,
                        windowsize: int):
    # Annoying napari induced warnings:
    import warnings
    warnings.filterwarnings("ignore")

    if channels is None:
        aprint("Channel(s) must be specified!")
        return

    if ':' not in input_path.replace("http://", ""):
        input_path = f'{input_path}:8000'

    with asection(f"Viewing remote dataset at: {input_path}, channel(s): {channels}"):
        channels = tuple(channel.strip() for channel in channels.split(','))
        channels = list(set(channels))

        from napari import Viewer, gui_qt

        with gui_qt():
            viewer = Viewer(title=f"DEXP | viewing with napari: {name} ", ndisplay=2)
            viewer.window.resize(windowsize + 256, windowsize)

            for channel in channels:
                import dask.array as da
                if '/' in channel:
                    array = da.from_zarr(f"{input_path}/{channel}")
                else:
                    array = da.from_zarr(f"{input_path}/{channel}/{channel}")

                if slicing is not None:
                    array = array[slicing]

                layer = viewer.add_image(array,
                                         name=channel,
                                         visible=True,
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

                try:
                    for axis in range(array.ndim):
                        if '/' in channel:
                            proj_array = da.from_zarr(f"{input_path}/{channel}_max{axis}")
                        else:
                            proj_array = da.from_zarr(f"{input_path}/{channel}/{channel}_max{axis}")

                        proj_layer = viewer.add_image(proj_array,
                                                      name=f'{channel}_max{axis}',
                                                      visible=True,
                                                      contrast_limits=contrast_limits,
                                                      blending='additive',
                                                      colormap=colormap)

                        if aspect is not None:
                            if axis == 0:
                                proj_layer.scale = (1, 1)
                            elif axis == 1:
                                proj_layer.scale = (aspect, 1)
                            elif axis == 2:
                                proj_layer.scale = (aspect, 1)

                            aprint(f"Setting aspect ratio for projection (layer.scale={proj_layer.scale})")

                except (KeyError, ArrayNotFoundError):
                    aprint("Warning: could not find projections in dataset!")
