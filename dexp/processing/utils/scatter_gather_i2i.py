from typing import Tuple, Union

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.utils.nd_slice import nd_split_slices, remove_margin_slice
from dexp.processing.utils.normalise import normalise_functions


def scatter_gather_i2i(function,
                       image,
                       chunks: Union[int, Tuple[int, ...]],
                       margins: Union[int, Tuple[int, ...]] = None,
                       normalise: bool = True,
                       clip: bool = False,
                       to_numpy: bool = True,
                       internal_dtype=None):
    """
    Image-2-image scatter-gather.
    'Scatters' computation of a given unary function by splitting the input array into chunks, computing using a given backend,
    and reassembling the chunks into a single array of same shape as the inpout that is either backed by the same backend than
    that of the input image, or that is backed by numpy -- usefull when the compute backend cannot hold the whole input and output
    images in memory.

    Parameters
    ----------
    function : unary function
    image : input image (can be any backend, numpy )
    chunks : chunk sizes to cut input image into, can be a single integer or a tuple of integers.
    margins : margins to add to each chunk, can be a single integer or a tuple of integers.
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

    if type(chunks) == int:
        chunks = (chunks,) * image.ndim

    if type(margins) == int:
        margins = (margins,) * image.ndim

    if to_numpy:
        result = numpy.empty(shape=image.shape, dtype=image.dtype)
    else:
        result = Backend.get_xp_module(image).empty_like(image)

    # Normalise:
    norm_fun, denorm_fun = normalise_functions(image, do_normalise=normalise, clip=clip)

    # image shape:
    shape = image.shape

    # We compute the slices objects to cut the input and target images into batches:
    chunk_slices = list(nd_split_slices(shape, chunks=chunks, margins=margins))
    chunk_slices_no_margins = list(nd_split_slices(shape, chunks=chunks))

    # Zipping together slices with and without margins:
    slices = zip(chunk_slices, chunk_slices_no_margins)

    # Number of tiles:
    number_of_tiles = len(chunk_slices)

    if number_of_tiles == 1:
        # If there is only one tile, let's not be complicated about it:
        result = denorm_fun(function(norm_fun(image)))
        if to_numpy:
            result = Backend.to_numpy(result, dtype=image.dtype)
        else:
            result = Backend.to_backend(result, dtype=image.dtype)
    else:
        _scatter_gather_loop(denorm_fun, function, image, internal_dtype, norm_fun, result, shape, slices, to_numpy)

    return result


def _scatter_gather_loop(denorm_fun, function, image, internal_dtype, norm_fun, result, shape, slices, to_numpy):
    for chunk_slice, chunk_slice_no_margins in slices:
        image_chunk = image[chunk_slice]
        image_chunk = Backend.to_backend(image_chunk, dtype=internal_dtype)
        image_chunk = denorm_fun(function(norm_fun(image_chunk)))
        if to_numpy:
            image_chunk = Backend.to_numpy(image_chunk, dtype=image.dtype)
        else:
            image_chunk = Backend.to_backend(image_chunk, dtype=image.dtype)

        remove_margin_slice_tuple = remove_margin_slice(
            shape, chunk_slice, chunk_slice_no_margins
        )
        image_chunk = image_chunk[remove_margin_slice_tuple]

        result[chunk_slice_no_margins] = image_chunk

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
