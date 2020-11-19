from pprint import pprint

import numpy
from skimage.data import camera

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.interpolation.warp import warp
from dexp.utils.timeit import timeit


def demo_warp_1d_numpy():
    try:
        backend = NumpyBackend()
        _demo_warp_1d(backend)
    except NotImplementedError:
        print("Numpy version not yet implemented")


def demo_warp_1d_cupy():
    try:
        backend = CupyBackend()
        _demo_warp_1d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_warp_1d(backend):

    image = numpy.random.uniform(low=0, high=1, size=(128,)).astype(dtype=numpy.float32)

    vector_field = numpy.random.uniform(low=-15, high=+15, size=(8,))

    with timeit("warp"):
        warped = warp(backend, image, vector_field)

    with timeit("dewarp"):
        dewarped = warp(backend, warped, -vector_field)

    warped = backend.to_numpy(warped)
    dewarped = backend.to_numpy(dewarped)

    pprint(warped-image)
    pprint(dewarped-image)

    pprint(numpy.max(numpy.absolute(warped-image)))
    pprint(numpy.max(numpy.absolute(dewarped-image)))


demo_warp_1d_cupy()
demo_warp_1d_numpy()
