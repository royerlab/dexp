import numpy

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.utils.timeit import timeit


def test_cupy_backend():
    try:
        available = CupyBackend.available_devices()
        print(f"Available devices: {available}")
        assert len(available) > 0

        for device_id in available:
            backend = CupyBackend(device_id)
            print(backend)

        array1 = numpy.random.uniform(0, 1, size=(512,) * 3).astype(numpy.float32)
        array2 = numpy.random.uniform(0, 1, size=(512,) * 3).astype(numpy.float32)

        backend = CupyBackend()

        array1 = backend.to_backend(array1, numpy.float32)
        array2 = backend.to_backend(array2, numpy.float32)

        with timeit("synchronise"):
            with backend:
                with timeit("gpu computation"):
                    def f(array1, array2):
                        for i in range(100):
                            array1 += array2
                            array2 /= (2 + array1)

                    backend.submit(f, array1, array2)

                backend.synchronise()

        # assert pytest.approx(array, rel=1e-5) == array_r
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")
