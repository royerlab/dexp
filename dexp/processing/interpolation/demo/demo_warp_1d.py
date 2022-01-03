from pprint import pprint

import numpy

from dexp.processing.interpolation.warp import warp
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_warp_1d_numpy():
    try:
        with NumpyBackend():
            _demo_warp_1d()
    except NotImplementedError:
        print("Numpy version not yet implemented")


def demo_warp_1d_cupy():
    try:
        with CupyBackend():
            _demo_warp_1d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_warp_1d():
    image = numpy.random.uniform(low=0, high=1, size=(128,)).astype(dtype=numpy.float32)

    magnitude = 15
    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(8,))

    with timeit("warp"):
        warped = warp(image, vector_field)

    with timeit("dewarp"):
        dewarped = warp(warped, -vector_field)

    warped = Backend.to_numpy(warped)
    dewarped = Backend.to_numpy(dewarped)

    pprint(warped - image)
    pprint(dewarped - image)

    pprint(numpy.max(numpy.absolute(warped - image)))
    pprint(numpy.max(numpy.absolute(dewarped - image)))


if __name__ == "__main__":
    demo_warp_1d_cupy()
    # demo_warp_1d_numpy()
