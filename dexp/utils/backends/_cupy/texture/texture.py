from typing import Tuple

import numpy

from dexp.utils.backends import Backend


def create_cuda_texture(
    array,
    texture_shape: Tuple[int, ...] = None,
    num_channels: int = 1,
    normalised_values: bool = False,
    normalised_coords: bool = False,
    sampling_mode: str = "linear",
    address_mode: str = "clamp",
    dtype=None,
):
    """Creates a Cuda texture and takes care of a lot of the needed 'know-how'

    Parameters
    ----------
    array
    texture_shape
    num_channels
    normalised_values
    normalised_coords
    sampling_mode
    address_mode
    dtype

    Returns
    -------

    """
    import cupy

    if texture_shape is None:
        if num_channels > 1:
            texture_shape = array.shape[0:-1]
        else:
            texture_shape = array.shape

    if array.dtype == numpy.float16 or dtype == numpy.float16:
        raise ValueError("float16 types not yet supported!")

    if dtype is None:
        dtype = array.dtype

    if not (1 <= len(texture_shape) <= 3):
        raise ValueError(
            f"Invalid number of dimensions ({len(texture_shape)}), must be 1, 2 or 3 (shape={texture_shape}) "
        )

    if not (num_channels == 1 or num_channels == 2 or num_channels == 4):
        raise ValueError(f"Invalid number of channels ({num_channels}), must be 1, 2., 3 or 4")

    if array.size != numpy.prod(texture_shape) * num_channels:
        raise ValueError(
            f"Texture shape {texture_shape}, num of channels ({num_channels}), "
            + f"and array size ({array.size}) are mismatched!"
        )

    dtype = numpy.dtype(dtype)

    if array.dtype != dtype:
        array = array.astype(dtype, copy=False)

    nbits = 8 * dtype.itemsize
    channels = (nbits,) * num_channels + (0,) * (4 - num_channels)
    if "f" in dtype.kind:
        channel_type = cupy.cuda.runtime.cudaChannelFormatKindFloat
    elif "i" in dtype.kind:
        channel_type = cupy.cuda.runtime.cudaChannelFormatKindSigned
    elif "u" in dtype.kind:
        channel_type = cupy.cuda.runtime.cudaChannelFormatKindUnsigned
    else:
        raise ValueError(f"dtype '{address_mode}' is not supported")

    format_descriptor = cupy.cuda.texture.ChannelFormatDescriptor(*channels, channel_type)

    cuda_array = cupy.cuda.texture.CUDAarray(format_descriptor, *(texture_shape[::-1]))
    ressource_descriptor = cupy.cuda.texture.ResourceDescriptor(
        cupy.cuda.runtime.cudaResourceTypeArray, cuArr=cuda_array
    )

    if address_mode == "clamp":
        address_mode = cupy.cuda.runtime.cudaAddressModeClamp
    elif address_mode == "border":
        address_mode = cupy.cuda.runtime.cudaAddressModeBorder
    elif address_mode == "wrap":
        address_mode = cupy.cuda.runtime.cudaAddressModeWrap
    elif address_mode == "mirror":
        address_mode = cupy.cuda.runtime.cudaAddressModeMirror
    else:
        raise ValueError(f"Address mode '{address_mode}' is not supported")

    address_mode = (address_mode,) * len(texture_shape)

    if sampling_mode == "nearest":
        filter_mode = cupy.cuda.runtime.cudaFilterModePoint
    elif sampling_mode == "linear":
        filter_mode = cupy.cuda.runtime.cudaFilterModeLinear
    else:
        raise ValueError(f"Sampling mode '{sampling_mode}' is not supported")

    if normalised_values:
        read_mode = cupy.cuda.runtime.cudaReadModeNormalizedFloat
    else:
        read_mode = cupy.cuda.runtime.cudaReadModeElementType

    texture_descriptor = cupy.cuda.texture.TextureDescriptor(
        addressModes=address_mode,
        filterMode=filter_mode,
        readMode=read_mode,
        sRGB=None,
        borderColors=None,
        normalizedCoords=normalised_coords,
        maxAnisotropy=None,
    )

    texture_object = cupy.cuda.texture.TextureObject(ressource_descriptor, texture_descriptor)

    # 'copy_from' from CUDAArray requires that the num of channels be multiplied
    # to the last axis of the array (see cupy docs!)
    if num_channels > 1:
        array_shape_for_copy = texture_shape[:-1] + (texture_shape[-1] * num_channels,)
    else:
        array_shape_for_copy = texture_shape
    axp = cupy.get_array_module(array)
    array = axp.reshape(array, newshape=array_shape_for_copy)

    if not array.flags.owndata or not not array.flags.c_contiguous:
        # the array must be contiguous, we check if this is a derived array,
        # if yes we must unfortunately copy the data...
        array = array.copy()

    # We need to synchronise otherwise some weird stuff happens! see warp 3d demo does not work withoutv this!
    Backend.current().synchronise()
    cuda_array.copy_from(array)

    del format_descriptor, texture_descriptor, ressource_descriptor

    return texture_object, cuda_array
