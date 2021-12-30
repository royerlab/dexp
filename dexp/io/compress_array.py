from math import ceil

import blosc
from blosc import compress_ptr, decompress_ptr, get_cbuffer_sizes
from numpy import ndarray


def compress_array(array: ndarray, clevel: int = 3, compressor: str = "lz4", min_num_chunks: int = 0) -> bytes:
    """
    Compresses an arbitrary ndarray that supports the '__array_interface__' into a Blosc compressed buffer.

    Parameters
    ----------
    array: Array to compress
    clevel: Compression level
    compressor: compressor (any supported by Blosc)
    min_num_chunks: Minimum number of chunks to split array into before compression.

    Returns
    -------
    Compressed buffer (as bytes)

    """
    array_address = array.__array_interface__["data"][0]
    array_length = array.nbytes
    end_address = array_address + array_length
    itemsize = array.itemsize

    # Blosc does not support 64bit buffers, only 32 bit buffers:
    max_chunk_length = blosc.MAX_BUFFERSIZE

    # Let's avoid small trailing chunks and make them all about teh same size:
    number_of_chunks = max(2, ceil(array_length / max_chunk_length)) if array_length > max_chunk_length else 1
    number_of_chunks = max(min_num_chunks, number_of_chunks)
    approx_chunk_length_in_bytes = array_length // number_of_chunks

    # Let's make sure that the chunks are aligned to the data type:
    approx_chunk_length_in_bytes = (approx_chunk_length_in_bytes // itemsize) * itemsize

    # Let' make the chunks large enough that we are garanteed to cover the whole array:
    approx_chunk_length_in_bytes += number_of_chunks * itemsize

    # this will hold the compressed data:
    compressed_buffer = bytearray()

    # we start reading the array data here:
    chunk_address = array_address

    # We loop through each chunk:
    while chunk_address < end_address:
        chunk_length_in_bytes = min(approx_chunk_length_in_bytes, end_address - chunk_address)
        compressed_chunk = compress_ptr(
            address=chunk_address,
            items=chunk_length_in_bytes // itemsize,
            typesize=itemsize,
            clevel=clevel,
            shuffle=blosc.BITSHUFFLE,
            cname=compressor,
        )
        if number_of_chunks == 1:
            # If there is only one chunk, let's not be complicated about it:
            compressed_buffer = compressed_chunk
        else:
            # If not, so bit it:
            compressed_buffer.extend(compressed_chunk)
        chunk_address += chunk_length_in_bytes

    return bytes(compressed_buffer)


def decompress_array(compressed_bytes, out_array: ndarray = None) -> ndarray:
    """
    Decompresses an array compressed with the 'compress_array' function.

    Parameters
    ----------
    compressed_bytes: buffer containing compressed data
    out_array: array of _correct_ size to put the decompressed daat in.

    Returns
    -------
    Same array as passed as 'out_array'

    """

    # get the array pointer and length in bytes:
    array_address = out_array.__array_interface__["data"][0]

    # prepare the basics:
    num_of_compressed_bytes = len(compressed_bytes)
    offset_compressed = 0
    address_decompressed = array_address

    while num_of_compressed_bytes - offset_compressed > 32:

        # This is the BLOSC header:
        blosc_header = bytes(compressed_bytes[offset_compressed : offset_compressed + 32])

        # we check how large is the first chunk available:
        num_decompressed_bytes, num_compressed_bytes, _ = get_cbuffer_sizes(blosc_header)

        # we prepare the corresponding region of the array:
        compressed_chunk = compressed_bytes[offset_compressed : offset_compressed + num_compressed_bytes]

        # do the actual decompression:
        decompress_ptr(compressed_chunk, address_decompressed)

        # we move the pointers forward:
        offset_compressed += num_compressed_bytes
        address_decompressed += num_decompressed_bytes

    return out_array
