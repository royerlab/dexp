import os
from os import listdir
from os.path import exists, isdir, isfile, join
from typing import Sequence, Tuple, Union

import imageio
from arbol.arbol import aprint, asection
from joblib import Parallel, delayed

from dexp.processing.color.blend import blend_color_images
from dexp.processing.color.insert import insert_color_image
from dexp.utils.backends import Backend, BestBackend


def blend_color_image_sequences(
    input_paths: Sequence[str],
    output_path: str = None,
    modes: Union[str, Sequence[str]] = "max",
    alphas: Union[float, Sequence[float]] = None,
    scales: Union[float, Sequence[float]] = None,
    translations: Union[str, Sequence[Tuple[Union[int, float], ...]]] = None,
    background_color: Tuple[float, float, float, float] = (0, 0, 0, 0),
    border_width: int = 1,
    border_color: Tuple[float, float, float, float] = None,
    border_over_image: bool = False,
    overwrite: bool = False,
    workers: int = -1,
    workersbackend: str = "threading",
    device: int = 0,
):
    """
    Blends several RGB(A) image sequences together

    Parameters
    ----------
    input_paths : Paths to folders containing images in some (lexicographic) order, or to single images
        (that will be broadcasted). If of different lengths the result's length is the shortest of the input sequences.
    output_path : Path to save the blended images.
    modes : Blending modes for each input sequence.
    alphas : Alpha transparency applied to each input sequence.
    scales : Scaling ('zoom' in scipy parlance) applied to each input sequence.
    translations : Translation applied to each input sequence.
    background_color:  Background color as tuple of normalised floats:  (R,G,B,A). Default is transparent black.
    border_width: Width of border added to insets.
    border_color: Border color.
    border_over_image: If True the border is not added but overlayed over the image, the image does not change size.
    overwrite : If True the output files are overwritten
    workers : Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)
    workersbackend : What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread)
    device: Device on  which to run the overlay computation (if a non-CPU device is available).
    """

    # ensure folder exists:
    os.makedirs(output_path, exist_ok=True)

    # collect all image files:
    image_sequences = []
    for input_path in input_paths:

        if isdir(input_path):
            # path is folder:
            pngfiles = [
                join(input_path, f) for f in listdir(input_path) if isfile(join(input_path, f)) and f.endswith(".png")
            ]
            pngfiles.sort()

        elif isfile(input_path) and (input_path.endswith("png") or input_path.endswith("jpg")):
            # path is image file:
            pngfiles = [
                input_path,
            ]

        image_sequences.append(pngfiles)

    # Basic sanity check on the images sequence: we determine the shortest non-one length :
    min_length = min(len(image_sequence) for image_sequence in image_sequences if len(image_sequence) != 1)
    max_length = max(len(image_sequence) for image_sequence in image_sequences if len(image_sequence) != 1)
    if min_length != max_length:
        aprint(f"Not all image sequences have the same non-one length: min:{min_length}, max:{max_length}")

    # Now we broadcast and crop in time:
    _image_sequences = []
    for image_sequence in image_sequences:
        if len(image_sequence) == 1:
            image_sequence = [
                image_sequence[0],
            ] * min_length
        elif len(image_sequence) > min_length:
            image_sequence = image_sequence[0:min_length]
        _image_sequences.append(image_sequence)
    image_sequences = _image_sequences
    nb_timepoints = min_length

    with BestBackend(device, exclusive=True, enable_unified_memory=True):

        def _process(backend: Backend, tp: int):

            with asection(f"processing time point: {tp}"):

                # Output file:
                filename = f"frame_{tp:05}.png"
                filepath = join(output_path, filename)

                # Write file:
                if overwrite or not exists(filepath):

                    # We set the backend of this thread to be the same as its parent thread:
                    if backend is not None:
                        Backend.set(backend)

                    # collect all images that need to be blended:
                    image_paths = list(image_sequence[tp] for image_sequence in image_sequences)

                    # Load images:
                    images = list(imageio.imread(image_path) for image_path in image_paths)

                    if scales is None and translations is None:
                        # Blend images:
                        blended = blend_color_images(
                            images=images, alphas=alphas, modes=modes, background_color=background_color
                        )
                    else:
                        xp = Backend.get_xp_module()

                        # Prepare background image:
                        blended = xp.zeros(shape=images[0].shape, dtype=images[0].dtype)

                        # Fill with background color:
                        for channel in range(4):
                            blended[:, channel] = background_color[channel]

                        # Insert each image at a different location with different blending
                        for inset_image, mode, scale, alpha, trans in zip(images, modes, scales, alphas, translations):
                            blended = insert_color_image(
                                image=blended,
                                inset_image=inset_image,
                                scale=scale,
                                translation=trans,
                                border_width=border_width,
                                border_color=border_color,
                                border_over_image=border_over_image,
                                mode=mode,
                                alpha=alpha,
                                background_color=(0, 0, 0, 0),
                            )

                    aprint(f"Writing file: {filename} in folder: {output_path}")
                    imageio.imwrite(filepath, Backend.to_numpy(blended), compress_level=1)
                else:
                    aprint(f"File: {filepath} already exists! use -w option to force overwrite...")

        with asection(f"Blending  {input_paths}, saving to {output_path}, for a total of {nb_timepoints} time points"):
            if workers == -1:
                workers = min(10, os.cpu_count() // 2)

            backend = Backend.current()

            if workers > 1:
                Parallel(n_jobs=workers, backend=workersbackend)(
                    delayed(_process)(backend, tp) for tp in range(nb_timepoints)
                )
            else:
                for tp in range(nb_timepoints):
                    _process(None, tp)

    aprint("Done!")
