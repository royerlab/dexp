import os
from os import listdir
from os.path import exists, isfile, join, isdir
from typing import Tuple, Sequence, Union

import imageio
from arbol.arbol import asection, aprint
from joblib import Parallel, delayed

from dexp.processing.color.blend import blend_color_images
from dexp.processing.color.scale_bar import insert_scale_bar
from dexp.processing.color.time_stamp import insert_time_stamp


def add_overlays_image_sequence(input_path: str,
                                output_path: str = None,
                                scale_bar: bool = True,
                                scale_bar_length_in_unit: float = 1,
                                scale_bar_pixel_scale: float = 1,
                                scale_bar_bar_height: int = 4,
                                scale_bar_translation: Union[str, Sequence[Tuple[Union[int, float], ...]]] = 'bottom_right',
                                scale_bar_unit: str = 'μm',
                                time_stamp: bool = True,
                                time_stamp_start_time: float = 0,
                                time_stamp_time_interval: float = 1,
                                time_stamp_translation: Union[str, Sequence[Tuple[Union[int, float], ...]]] = 'top_right',
                                time_stamp_unit: str = 's',
                                margin: float = 1,
                                color: Tuple[float, float, float, float] = None,
                                number_format: str = '{:.1f}',
                                font_name: str = "Helvetica",
                                font_size: float = 32,
                                mode: str = 'max',
                                overwrite: bool = False,
                                workers: int = -1,
                                workersbackend: str = 'threading'):
    """
    Blends several RGB(A) image sequences together

    Parameters
    ----------
    input_path : Path to folder containing images in some (lexicographic) order.
    output_path : Path to save the blended images.
    scale_bar: True to insert scale bar.
    scale_bar_length_in_unit: Length of scale bar in the provided unit.
    scale_bar_pixel_scale: conversion factor from pixels to units -- what is the side length of a pixel/voxel in units.
    scale_bar_bar_height: Height of th scale bar in pixels
    scale_bar_translation: Positions of the scale bar in pixels in natural order: (x, y). Can also be a string: 'bottom_left', 'bottom_right', 'top_left', 'top_right'.
    scale_bar_unit: Scale bar unit name.
    time_stamp: True to insert time stamp.
    time_stamp_start_time: Start time for time stamp
    time_stamp_time_interval: Time interval inn units of time between consecutive images.
    time_stamp_translation: Positions of the time stamp in pixels in natural order: (x, y). Can also be a string: 'bottom_left', 'bottom_right', 'top_left', 'top_right'.
    time_stamp_unit: Time stamp time unit name.
    margin: margin around bar expressed in units relative to the text height
    color: Color of the bar and text as tuple of 4 values: (R, G, B, A)
    number_format: Format string to represent the start and end values.
    font_name: Font name.
    font_size: Font size in pixels.
    mode: Blending mode. See function 'blend_color_images' for available blending modes.
    overwrite : If True the output files are overwritten
    workers : Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)
    workersbackend : What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread)
    """

    # ensure folder exists:
    os.makedirs(output_path, exist_ok=True)

    # collect image files:
    if isdir(input_path):
        # path is folder:
        png_file_paths = [join(input_path, f) for f in listdir(input_path) if isfile(join(input_path, f)) and f.endswith('.png')]
        png_file_paths.sort()
    else:
        raise ValueError("Input path must be folder containing at least one image")

    # Load images:
    with asection("Load images..."):
        images = list(imageio.imread(p) for p in png_file_paths)

    # Apply time stamp:
    if time_stamp:
        with asection("Apply time stamp..."):
            images = insert_time_stamp(images=images,
                                       start_time=time_stamp_start_time,
                                       time_interval=time_stamp_time_interval,
                                       unit=time_stamp_unit,
                                       margin=margin,
                                       translation=time_stamp_translation,
                                       color=color,
                                       number_format=number_format,
                                       font_name=font_name,
                                       font_size=font_size,
                                       mode=mode)

    if scale_bar:

        # First generate the scale bar itself:

        _, scale_bar_image = insert_scale_bar(images[0],
                                              length_in_unit=scale_bar_length_in_unit,
                                              pixel_scale=scale_bar_pixel_scale,
                                              bar_height=scale_bar_bar_height,
                                              margin=margin,
                                              translation=scale_bar_translation,
                                              color=color,
                                              number_format=number_format,
                                              font_name=font_name,
                                              font_size=font_size,
                                              unit=scale_bar_unit,
                                              mode=mode
                                              )

        # The number of time points is the number of images in the sequence:
        nb_timepoints = len(images)

        def _process(tp: int):

            with asection(f'processing time point: {tp}'):

                # get image:
                image = images[tp]

                # Blend images:
                image_with_scale_bar = blend_color_images(images=(image,
                                                                  scale_bar_image),
                                                          alphas=(1, 1),
                                                          modes=('max', mode))

                # Output file:
                filename = f"frame_{tp:05}.png"
                filepath = join(output_path, filename)

                # Write file:
                if overwrite or not exists(filepath):
                    aprint(f"Writing file: {filename} in folder: {output_path}")
                    imageio.imwrite(filepath, image_with_scale_bar)
                else:
                    aprint(f"File: {filepath} already exists! use -w option to force overwrite...")

        with asection(f"Adding time-stamp ({insert_time_stamp}) and scale-bar ({insert_scale_bar}) to: {input_path}, and saving to {output_path}, for a total of {nb_timepoints} time points"):
            if workers <= 0:
                workers = os.cpu_count() // 2

            Parallel(n_jobs=workers, backend=workersbackend)(delayed(_process)(tp) for tp in range(nb_timepoints))
            aprint(f"Done!")
