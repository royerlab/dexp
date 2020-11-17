from typing import Tuple

import numpy
import cupy


def create_cuda_texture(shape:Tuple[int,...], num_channels:int = 1, sampling_mode='linear', address_mode='clamp', dtype=numpy.float16):

    if not  1<=len(shape)<=3:
        raise ValueError(f"Invalid number of dimensions ({len(shape)}), must be 1, 2 or 3 (shape={shape}) ")

    if not  1<=num_channels<=4:
        raise ValueError(f"Invalid number of channels ({num_channels}), must be 1, 2., 3 or 4")




    nbits = 8*dtype.nbytes
    channels = (nbits,)*num_channels+(0,)*(4-num_channels)
    if 'f' in dtype.type:
        channel_type = cupy.cuda.runtime.cudaChannelFormatKindFloat
    elif 'i' in dtype.type:
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
        address_mode = cupy.cuda.runtime.cudaAddressModeWrap

    address_mode = (address_mode,) * len(shape)

    if sampling_mode == 'nearest':
        filter_mode = cupy.cuda.runtime.cudaFilterModePoint
    elif sampling_mode == 'linear':
        filter_mode = cupy.cuda.runtime.cudaFilterModeLinear
    else:
        raise ValueError(f"Sampling mode '{sampling_mode}' not supported")

    texture_descriptor = cupy.cuda.texture.TextureDescriptor(address_mode,
                                                             filter_mode,
                                                             cupy.cuda.runtime.cudaReadModeElementType)


    texture_object = cupy.cuda.texture.TextureObject(ressource_descriptor,
                                             texture_descriptor)

    return texture_object, cuda_array