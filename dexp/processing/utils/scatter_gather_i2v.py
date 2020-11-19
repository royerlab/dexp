import math
from typing import Tuple, Union, Sequence, Any

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.utils.nd_slice import nd_split_slices, remove_margin_slice

def scatter_gather_i2v(backend: Backend,
                       function,
                       images: Union[Any, Tuple[Any]],
                       chunks: Union[int, Tuple[int, ...]],
                       margins: Union[int, Tuple[int, ...]] = None,
                       to_numpy: bool = True,
                       internal_dtype=None):
    """
    Image-2-vector scatter-gather.
    'Scatters' computation of a given unary function by splitting the input array into chunks, computing using a given backend,
    and reassembling the chunks into a single array of vectors, with one vector per chunk.

    Parameters
    ----------
    backend : Backend to use for computation
    function : unary function
    images : sequence of images (can be any backend, numpy )
    chunks : chunks to cut input image into, can be a single integer or a tuple of integers.
    margins : margins to add to each chunk, can be a single integer or a tuple of integers.
    to_numpy : should the result be a numpy array? Very usefull when the compute backend cannot hold the whole input and output images in memory.
    internal_dtype : internal dtype for computation

    Returns
    -------
    Result of applying the unary function to the input image, if to_numpy==True then the image is

    """
    xp = backend.get_xp_module()

    if type(images) is not tuple:
        images = (images,)

    for image in images:
        if image.shape!=images[0].shape:
            raise ValueError("All images must have the same shape!")

    first_image = images[0]
    ndim = first_image.ndim
    shape = first_image.shape
    dtype = first_image.dtype

    if internal_dtype is None:
        internal_dtype = dtype

    if type(chunks) == int:
        chunks = (chunks,) * ndim

    if type(margins) == int:
        margins = (margins,) * ndim

    # We compute the slices objects to cut the input image into batches:
    chunk_slices = list(nd_split_slices(shape, chunks=chunks, margins=margins))
    chunk_slices_no_margins = list(nd_split_slices(shape, chunks=chunks))

    # We compute the shape at the chunk level:
    chunk_shape = tuple(math.ceil(s/c) for s, c in zip(shape,chunks))

    # Zipping together slices with and without margins:
    slices = zip(chunk_slices, chunk_slices_no_margins)

    # Number of tiles:
    number_of_tiles = len(chunk_slices)

    if number_of_tiles == 1:
        # If there is only one tile, let's not be complicated about it:
        result = function(*images)
        if to_numpy:
            result = backend.to_numpy(result, dtype=dtype)
        else:
            result = backend.to_backend(result, dtype=dtype)
        result = xp.reshape(result, newshape=(1,)*ndim + result.shape)
    else:
        result_list = []
        for chunk_slice, chunk_slice_no_margins in slices:
            image_chunks = tuple(image[chunk_slice] for image in images)
            image_chunks = tuple(backend.to_backend(image_chunk, dtype=internal_dtype) for image_chunk in image_chunks)
            result = function(*image_chunks)
            if to_numpy:
                result = backend.to_numpy(result, dtype=image.dtype)
            else:
                result = backend.to_backend(result, dtype=image.dtype)
            result_list.append(result)

        rxp = backend.get_xp_module(result_list[0])
        result = rxp.stack(result_list)
        result = rxp.reshape(result, newshape=chunk_shape+result_list[0].shape)

    return result


