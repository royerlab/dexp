from typing import Tuple

import numpy
import cupy


def create_cuda_texture(array,
                        shape:Tuple[int,...]=None,
                        num_channels:int = 1,
                        normalised_values: bool = False,
                        normalised_coords: bool = False,
                        sampling_mode: str ='linear',
                        address_mode: str ='clamp',
                        dtype=None):


    if not  1<=len(shape)<=3:
        raise ValueError(f"Invalid number of dimensions ({len(shape)}), must be 1, 2 or 3 (shape={shape}) ")

    if not  1<=num_channels<=4:
        raise ValueError(f"Invalid number of channels ({num_channels}), must be 1, 2., 3 or 4")

    if dtype is None:
        dtype=array.dtype

    dtype = numpy.dtype(dtype)

    if array.dtype != dtype:
        array = array.astype(dtype, copy=False)

    nbits = 8*dtype.itemsize
    channels = (nbits,)*num_channels+(0,)*(4-num_channels)
    if 'f' in dtype.kind:
        channel_type = cupy.cuda.runtime.cudaChannelFormatKindFloat
    elif 'i' in dtype.kind:
        channel_type = cupy.cuda.runtime.cudaChannelFormatKindInt

    format_descriptor = cupy.cuda.texture.ChannelFormatDescriptor(*channels, channel_type)

    cuda_array = cupy.cuda.texture.CUDAarray(format_descriptor, *shape)
    ressource_descriptor  = cupy.cuda.texture.ResourceDescriptor(cupy.cuda.runtime.cudaResourceTypeArray, cuArr=cuda_array)

    if address_mode=='clamp':
        address_mode = cupy.cuda.runtime.cudaAddressModeClamp
    elif address_mode=='border':
        address_mode = cupy.cuda.runtime.cudaAddressModeBorder
    elif address_mode == 'wrap':
        address_mode = cupy.cuda.runtime.cudaAddressModeWrap
    elif address_mode == 'mirror':
        address_mode = cupy.cuda.runtime.cudaAddressModeMirror
    else:
        raise ValueError(f"Address mode '{address_mode}' not supported")

    address_mode = (address_mode,) * len(shape)

    if sampling_mode == 'nearest':
        filter_mode = cupy.cuda.runtime.cudaFilterModePoint
    elif sampling_mode == 'linear':
        filter_mode = cupy.cuda.runtime.cudaFilterModeLinear
    else:
        raise ValueError(f"Sampling mode '{sampling_mode}' not supported")

    if normalised_values:
        read_mode = cupy.cuda.runtime.cudaReadModeNormalizedFloat
    else:
        read_mode = cupy.cuda.runtime.cudaReadModeElementType

    texture_descriptor = cupy.cuda.texture.TextureDescriptor(addressModes=address_mode,
                                                             filterMode=filter_mode,
                                                             readMode=read_mode,
                                                             sRGB=None,
                                                             borderColors=None,
                                                             normalizedCoords=normalised_coords,
                                                             maxAnisotropy=None)

    texture_object = cupy.cuda.texture.TextureObject(ressource_descriptor,
                                                     texture_descriptor)

    cuda_array.copy_from(array)

    return texture_object