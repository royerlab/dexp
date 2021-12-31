import os
from os import listdir
from os.path import exists, isdir, isfile, join
from typing import Tuple, Union

import imageio
from arbol.arbol import aprint, asection
from joblib import Parallel, delayed

from dexp.processing.color.crop_resize_pad import crop_resize_pad_color_image
from dexp.utils.backends import Backend, BestBackend, CupyBackend


def crop_resize_pad_image_sequence(
    input_path: str,
    output_path: str = None,
    crop: Union[int, Tuple[int, ...], Tuple[Tuple[int, int], ...]] = None,
    resize: Tuple[int, ...] = None,
    resize_order: int = 3,
    resize_mode: str = "constant",
    pad_width: Tuple[Tuple[int, int], ...] = None,
    pad_mode: str = "constant",
    pad_color: Tuple[float, float, float, float] = (0, 0, 0, 0),
    rgba_value_max: float = 255,
    overwrite: bool = False,
    workers: int = -1,
    workersbackend: str = "threading",
    device: int = 0,
):
    """
    Crops, resizes and then pad a sequence of RGB(A) images.

    Parameters
    ----------
    input_path : Path to folder containing images in some (lexicographic) order.
    output_path : Path to save the blended images.
    crop: Crop image by removing a given number of pixels/voxels per axis.
        For example: ((10,20),(10,20)) crops 10 pixels on the left for axis 0,
        20 pixels from the right of axis 0, and the same for axis 2.
    resize: After cropping, the image is resized to the given shape. If any entry in the tuple is -1 then
        that position in the shape is automatically determined based on the existing shape to preserve aspect ratio.
    resize_order: The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
    resize_mode: optional The mode parameter determines how the input array is extended beyond its boundaries.
        Can be: ‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’.
    pad_width: After cropping and resizing, padding is performed.
        The provided tuple is interpreted similarly to cropping.
    pad_mode: Padding mode, see numpy.pad for the available modes.
    pad_color: Padding color as tuple of normalised floats:  (R,G,B,A). Default is transparent black.
    rgba_value_max: max value for rgba values.
    overwrite : If True the output files are overwritten
    workers : Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)
    workersbackend : What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread)
    device: Device on  which to run the overlay computation (if a non-CPU device is available).
    """

    # accomodate cupy limits:
    if type(Backend.current()) == CupyBackend:
        resize_order = min(resize_order, 1)

    # ensure folder exists:
    os.makedirs(output_path, exist_ok=True)

    # collect image files:
    if isdir(input_path):
        # path is folder:
        png_file_paths = [
            join(input_path, f) for f in listdir(input_path) if isfile(join(input_path, f)) and f.endswith(".png")
        ]
        png_file_paths.sort()
    else:
        raise ValueError("Input path must be folder containing at least one image")

    if workers == -1:
        workers = min(8, os.cpu_count() // 2)

    # Number of timepoints:
    nb_timepoints = len(png_file_paths)

    def _process(tp: int):
        with asection(f"Processing time point: {tp}"):
            with BestBackend(device, exclusive=True, enable_unified_memory=True):

                # Output file:
                filename = f"frame_{tp:05}.png"
                filepath = join(output_path, filename)

                # Write file:
                if overwrite or not exists(filepath):

                    # Get image path:
                    image_path = png_file_paths[tp]

                    # get image:
                    image = imageio.imread(image_path)

                    resized_image = crop_resize_pad_color_image(
                        image=image,
                        crop=crop,
                        resize=resize,
                        resize_order=resize_order,
                        resize_mode=resize_mode,
                        pad_width=pad_width,
                        pad_mode=pad_mode,
                        pad_color=pad_color,
                        rgba_value_max=rgba_value_max,
                    )

                    with asection(f"Writing file: {filename} in folder: {output_path}"):
                        imageio.imwrite(filepath, Backend.to_numpy(resized_image), compress_level=1)
                else:
                    aprint(f"File: {filepath} already exists! use -w option to force overwrite...")

    with asection(
        f"Cropping ({crop}), resizing ({resize}), padding ({pad_width}), images at: {input_path}, "
        + f"and saving to {output_path}, for a total of {nb_timepoints} time points"
    ):
        Parallel(n_jobs=workers, backend=workersbackend)(delayed(_process)(tp) for tp in range(nb_timepoints))
        aprint("Done!")
