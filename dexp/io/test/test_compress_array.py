import numpy

from dexp.io.compress_array import compress_array, decompress_array


def do_test(array_dc, array_uc, min_num_chunks=0):
    compressed_array = compress_array(array_uc,
                                      min_num_chunks=min_num_chunks)
    ratio = len(compressed_array) / (array_uc.size * array_uc.itemsize)
    print("Compression succeeded!")
    print(f"Compression ratio: {ratio}")
    decompress_array(compressed_array, array_dc)
    print("Decompression succeeded!")
    assert (array_uc == array_dc).all()

# too big, it generates out-of-memory error
# def test_compress_huge_array():
#     array_uc = numpy.linspace(0, 1, 2**33).astype(numpy.uint16)
#     array_dc = numpy.empty_like(array_uc)
#     do_test(array_dc, array_uc)
#
# def test_compress_large_array():
#     array_uc = numpy.linspace(0, 1024, 2**30).astype(numpy.uint16)
#     array_dc = numpy.empty_like(array_uc)
#     do_test(array_dc, array_uc)

def test_compress_small_array():
    array_uc = numpy.linspace(0, 1024, 2**15).astype(numpy.uint16)
    array_dc = numpy.empty_like(array_uc)
    do_test(array_dc, array_uc)

def test_compress_tiny_array():
    array_uc = numpy.linspace(0, 1024, 1)
    array_dc = numpy.empty_like(array_uc).astype(numpy.uint16)
    do_test(array_dc, array_uc)

def test_compress_many_chunks():
    array_uc = numpy.linspace(0, 1024, 1000).astype(numpy.uint16)
    array_dc = numpy.empty_like(array_uc)
    do_test(array_dc, array_uc, min_num_chunks=987)


