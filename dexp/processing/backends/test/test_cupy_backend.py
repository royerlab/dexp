import time

import numpy
from joblib import Parallel, delayed

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend


def test_cupy_basics():
    try:
        import cupy
        with cupy.cuda.Device(1):
            array = cupy.array([1, 2, 3])
            print("\nWorked!")

    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def test_list_devices():
    try:
        # Test list devices:

        available = CupyBackend.available_devices()
        print(f"Available devices: {available}")
        assert len(available) > 0

        for device_id in available:
            with CupyBackend(device_id) as backend:
                print(backend)
                xp = Backend.get_xp_module()

                array1 = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)
                array2 = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)

                array1 = Backend.to_backend(array1, numpy.float32)
                array2 = Backend.to_backend(array2, numpy.float32)

    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def test_allocation_pool():
    try:

        with CupyBackend(enable_memory_pool=True) as backend:
            backend.clear_allocation_pool()

            xp = backend.get_xp_module()

            import cupy
            mempool = cupy.get_default_memory_pool()
            in_pool_before = mempool.total_bytes() - mempool.used_bytes()
            print(f"in_pool_before={in_pool_before}")

            for i in range(10):
                array = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)

            time.sleep(2)
            backend.synchronise()

            in_pool_after = mempool.total_bytes() - mempool.used_bytes()
            print(f"in_pool_after={in_pool_after}")

            assert in_pool_after > in_pool_before

            backend.clear_allocation_pool()

            in_pool_after_clear = mempool.total_bytes() - mempool.used_bytes()
            print(f"in_pool_after_clear={in_pool_after_clear}")

            assert in_pool_after_clear <= in_pool_before

            for i in range(10):
                array = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)

            in_pool_after_reallocation = mempool.total_bytes() - mempool.used_bytes()
            print(f"in_pool_after_reallocation={in_pool_after_reallocation}")

            assert in_pool_after_reallocation > in_pool_after_clear

        in_pool_after_context = mempool.total_bytes() - mempool.used_bytes()
        print(f"in_pool_after_context={in_pool_after_clear}")

        assert in_pool_after_context == in_pool_after_clear


    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def test_paralell():
    try:

        def f(id):
            with CupyBackend(id):
                print(f"Begin: Job on device #{id}")
                xp = Backend.get_xp_module()
                array = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)
                array += 1
                print(f"End: Job on device #{id}")

        n_jobs = 2
        Parallel(n_jobs=n_jobs, backend='threading')(delayed(f)(id) for id in range(n_jobs))

    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")
