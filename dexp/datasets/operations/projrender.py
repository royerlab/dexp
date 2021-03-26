from os import makedirs
from os.path import exists, join
from typing import Any, Tuple, Sequence

import imageio
from arbol.arbol import aprint, asection
from joblib import delayed, Parallel

from dexp.datasets.base_dataset import BaseDataset
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.best_backend import BestBackend
from dexp.processing.color.projection import project_image


def dataset_projection_rendering(input_dataset: BaseDataset,
                                 output_path: str = None,
                                 channels: Sequence[str] = None,
                                 slicing: Any = None,
                                 overwrite: bool = False,
                                 axis: int = 0,
                                 dir: int = -1,
                                 mode: str = 'colormax',
                                 clim: Tuple[float, float] = None,
                                 attenuation: float = 0.05,
                                 gamma: float = 1.0,
                                 dlim: Tuple[float, float] = None,
                                 colormap: str = None,
                                 rgbgamma: float = 1.0,
                                 transparency: bool = False,
                                 legendsize: float = 1.0,
                                 legendscale: float = 1.0,
                                 legendtitle: str = 'color-coded depth (voxels)',
                                 legendtitlecolor: Tuple[float, float, float, float] = (1, 1, 1, 1),
                                 legendposition: str = 'bottom_left',
                                 legendalpha: float = 1.0,
                                 step: int = 1,
                                 workers: int = -1,
                                 workersbackend: str = 'threading',
                                 devices: Tuple[int, ...] = (0,),
                                 stop_at_exception=True):
    for channel in channels:

        # Ensures that the output folder exists per channel:
        if len(channels) == 1:
            channel_output_path = output_path
        else:
            channel_output_path = output_path + f'_{channel}'

        makedirs(channel_output_path, exist_ok=True)

        with asection(f"Channel '{channel}' shape: {input_dataset.shape(channel)}:"):
            aprint(input_dataset.info(channel))

        array = input_dataset.get_array(channel, wrap_with_dask=True)

        if slicing:
            array = array[slicing]

        aprint(f"Rendering array of shape={array.shape} and dtype={array.dtype} for channel '{channel}'.")

        nbframes = array.shape[0]

        with asection("Rendering:"):

            def process(tp, _clim, device):
                try:
                    with asection(f"Rendering Frame     : {tp:05}"):

                        filename = join(channel_output_path, f"frame_{tp:05}.png")

                        if overwrite or not exists(filename):

                            with asection("Loading stack..."):
                                stack = array[tp].compute()

                            with BestBackend(device, exclusive=True, enable_unified_memory=True):
                                if _clim is not None:
                                    aprint(f"Using provided min and max for contrast limits: {_clim}")
                                    min_value, max_value = (float(strvalue) for strvalue in _clim.split(','))
                                    _clim = (min_value, max_value)

                                with asection(f"Projecting image of shape: {stack.shape} "):
                                    projection = project_image(stack,
                                                               axis=axis,
                                                               dir=dir,
                                                               mode=mode,
                                                               attenuation=attenuation,
                                                               attenuation_min_density=0.002,
                                                               attenuation_filtering=4,
                                                               gamma=gamma,
                                                               clim=_clim,
                                                               cmap=colormap,
                                                               dlim=dlim,
                                                               rgb_gamma=rgbgamma,
                                                               transparency=transparency,
                                                               legend_size=legendsize,
                                                               legend_scale=legendscale,
                                                               legend_title=legendtitle,
                                                               legend_title_color=legendtitlecolor,
                                                               legend_position=legendposition,
                                                               legend_alpha=legendalpha)

                                with asection(f"Saving frame {tp} as: {filename}"):
                                    imageio.imwrite(filename,
                                                    Backend.to_numpy(projection),
                                                    compress_level=1)

                except Exception as error:
                    aprint(error)
                    aprint(f"Error occurred while processing time point {tp} !")
                    import traceback
                    traceback.print_exc()

                    if stop_at_exception:
                        raise error

            if workers == -1:
                workers = len(devices)
            aprint(f"Number of workers: {workers}")

            if workers > 1:
                Parallel(n_jobs=workers, backend=workersbackend)(delayed(process)(tp, clim, devices[tp % len(devices)]) for tp in range(0, nbframes, step))
            else:
                for tp in range(0, nbframes, step):
                    process(tp, clim, devices[0])
