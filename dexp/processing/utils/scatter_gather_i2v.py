import math
from typing import Callable, Optional, Tuple, Union

import numpy as np

from dexp.processing.utils.nd_slice import nd_split_slices
from dexp.utils import xpArray
from dexp.utils.backends import Backend


def scatter_gather_i2v(
    images: Union[xpArray, Tuple[xpArray]],
    function: Callable,
    tiles: Union[int, Tuple[int, ...]],
    margins: Optional[Union[int, Tuple[int, ...]]] = None,
    to_numpy: bool = True,
    internal_dtype: Optional[np.dtype] = None,
) -> xpArray:
    """
    Image-2-vector scatter-gather.
    Given a n-ary function that takes n images and returns a tuple of vectors per image,
    we split the input arrays into tiles, apply the function to each tile computing using a given backend, and
    reassembling the vectors into a tuple of arrays of vectors, with one vector per tile.

    Parameters
    ----------
    images : sequence of images (can be any backend, numpy)
    function : n-ary function. Must accept one or more arrays -- of same shape
        --- and return a _tuple_ of arrays as result.
    tiles : tiles to cut input image into, can be a single integer or a tuple of integers.
    margins : margins to add to each tile, can be a single integer or a tuple of integers.
    to_numpy : should the result be a numpy array? Very usefull when the compute backend
        cannot hold the whole input and output images in memory.
    internal_dtype : internal dtype for computation

    Returns
    -------
    Result a tuple of arrays, each array having the dimension of the input images + an extra 'vector' dimensions.
    The shape of the arrays matches the tiling of the input images. Each vector is the result of applying the
    function to the input images. If to_numpy==True then the returned arrays are numpy array.

    """

    xp = Backend.get_xp_module()

    if type(images) is not tuple:
        images = (images,)

    for image in images:
        if image.shape != images[0].shape:
            raise ValueError("All images must have the same shape!")

    first_image = images[0]
    ndim = first_image.ndim
    shape = first_image.shape
    dtype = first_image.dtype

    if internal_dtype is None:
        internal_dtype = dtype

    if type(tiles) == int:
        tiles = (tiles,) * ndim

    if margins is None:
        margins = (0,) * ndim
    elif type(margins) == int:
        margins = (margins,) * ndim

    # We compute the slices objects to cut the input image into batches:
    tile_slices = list(nd_split_slices(shape, chunks=tiles, margins=margins))
    tile_slices_no_margins = list(nd_split_slices(shape, chunks=tiles))

    # We compute the shape at the tile level:
    tile_shape = tuple(math.ceil(s / c) for s, c in zip(shape, tiles))

    # Zipping together slices with and without margins:
    slices = zip(tile_slices, tile_slices_no_margins)

    # Number of tiles:
    number_of_tiles = len(tile_slices)

    if number_of_tiles == 1:
        # If there is only one tile, let's not be complicated about it:
        results = function(*images)
        if to_numpy:
            results = tuple(Backend.to_numpy(result, dtype=dtype) for result in results)
        else:
            results = tuple(Backend.to_backend(result, dtype=dtype) for result in results)
        results_stacked_reshaped = tuple(xp.reshape(result, newshape=(1,) * ndim + result.shape) for result in results)
    else:
        results_lists = None
        for tile_slice, tile_slice_no_margins in slices:
            image_tiles = tuple(image[tile_slice] for image in images)
            image_tiles = tuple(Backend.to_backend(image_tile, dtype=internal_dtype) for image_tile in image_tiles)
            results = function(*image_tiles)
            if to_numpy:
                results = tuple(Backend.to_numpy(result, dtype=image.dtype) for result in results)
            else:
                results = tuple(Backend.to_backend(result, dtype=image.dtype) for result in results)

            if results_lists is None:
                results_lists = tuple([] for _ in results)

            for result, results_list in zip(results, results_lists):
                results_list.append(result)

        rxps = tuple(Backend.get_xp_module(results_list[0]) for results_list in results_lists)
        results_stacked = tuple(rxp.stack(results_list) for rxp, results_list in zip(rxps, results_lists))
        results_stacked_reshaped = tuple(
            rxp.reshape(stack, newshape=tile_shape + results[0].shape)
            for rxp, stack, results in zip(rxps, results_stacked, results_lists)
        )

    return results_stacked_reshaped
