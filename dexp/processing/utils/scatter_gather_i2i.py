from typing import Tuple, Union, Optional

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.utils.nd_slice import nd_split_slices, remove_margin_slice
from dexp.processing.utils.normalise import normalise_functions


def scatter_gather_i2i(function,
                       image,
                       tiles: Union[int, Tuple[int, ...]],
                       margins: Optional[Union[int, Tuple[int, ...]]] = None,
                       normalise: bool = False,
                       clip: bool = False,
                       to_numpy: bool = True,
                       internal_dtype=None):
    """
    Image-2-image scatter-gather.
    'Scatters' computation of a given unary function by splitting the input array into tiles, computing using a given backend,
    and reassembling the tiles into a single array of same shape as the inpout that is either backed by the same backend than
    that of the input image, or that is backed by numpy -- usefull when the compute backend cannot hold the whole input and output
    images in memory.

    Parameters
    ----------
    function : unary function
    image : input image (can be any backend, numpy )
    tiles : tile sizes to cut input image into, can be a single integer or a tuple of integers.
    margins : margins to add to each tile, can be a single integer or a tuple of integers. if None, no margins are added.
    normalise : normalises  the input image.
    clip : clip after normalisation/denormalisation
    to_numpy : should the result be a numpy array? Very usefull when the compute backend cannot hold the whole input and output images in memory.
    internal_dtype : internal dtype for computation

    Returns
    -------
    Result of applying the unary function to the input image, if to_numpy==True then the image is

    """

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(tiles) == int:
        tiles = (tiles,) * image.ndim

    # If None is passed for a tile that means that we don't tile along that axis, we als clip the tile size:
    tiles = tuple((length if tile is None else min(length, tile)) for tile, length in zip(tiles, image.shape))

    if margins is None:
        margins = (0,) * image.ndim

    if type(margins) == int:
        margins = (margins,) * image.ndim

    if to_numpy:
        result = numpy.empty(shape=image.shape, dtype=internal_dtype)
    else:
        result = Backend.get_xp_module(image).empty_like(image, dtype=internal_dtype)

    # Normalise:
    norm_fun, denorm_fun = normalise_functions(
        Backend.to_backend(image),
        do_normalise=normalise, clip=clip,
        quantile=0.005
    )

    # image shape:
    shape = image.shape

    # We compute the slices objects to cut the input and target images into batches:
    tile_slices = list(nd_split_slices(shape, chunks=tiles, margins=margins))
    tile_slices_no_margins = list(nd_split_slices(shape, chunks=tiles))

    # Zipping together slices with and without margins:
    slices = zip(tile_slices, tile_slices_no_margins)

    # Number of tiles:
    number_of_tiles = len(tile_slices)

    if number_of_tiles == 1:
        # If there is only one tile, let's not be complicated about it:
        result = denorm_fun(function(norm_fun(image)))
        if to_numpy:
            result = Backend.to_numpy(result, dtype=internal_dtype)
        else:
            result = Backend.to_backend(result, dtype=internal_dtype)
    else:
        _scatter_gather_loop(denorm_fun, function, image, internal_dtype, norm_fun, result, shape, slices, to_numpy)

    return result


def _scatter_gather_loop(denorm_fun, function, image, internal_dtype, norm_fun, result, shape, slices, to_numpy):
    for tile_slice, tile_slice_no_margins in slices:
        image_tile = image[tile_slice]
        image_tile = Backend.to_backend(image_tile, dtype=internal_dtype)
        image_tile = denorm_fun(function(norm_fun(image_tile)))
        if to_numpy:
            image_tile = Backend.to_numpy(image_tile, dtype=internal_dtype)
        else:
            image_tile = Backend.to_backend(image_tile, dtype=internal_dtype)

        remove_margin_slice_tuple = remove_margin_slice(
            shape, tile_slice, tile_slice_no_margins
        )
        image_tile = image_tile[remove_margin_slice_tuple]

        result[tile_slice_no_margins] = image_tile

# Dask turned out not too work great here, HUGE overhead compared to the light approach above.
# def scatter_gather_dask(backend: Backend,
#                         function,
#                         image,
#                         chunks,
#                         margins=None):
#     boundary=None
#     trim=True
#     align_arrays=True
#
#     image_d = from_array(image, chunks=chunks, asarray=False)
#
#     def function_numpy(_image):
#         print(_image.shape)
#         return backend.to_numpy(function(_image))
#
#     #func, *args, depth=None, boundary=None, trim=True, align_arrays=True, **kwargs
#     computation= map_overlap(function_numpy,
#                 image_d,
#                 depth=margins,
#                 boundary=boundary,
#                 trim=trim,
#                 align_arrays=align_arrays,
#                 dtype=image.dtype
#                 )
#
#     #computation.visualize(filename='transpose.png')
#     result = computation.compute()
#
#     return result
