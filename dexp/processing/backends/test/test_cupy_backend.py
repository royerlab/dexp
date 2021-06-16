import gc
from functools import reduce
from time import time, sleep

import numpy
from arbol import aprint, asection
from joblib import Parallel, delayed

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.interpolation.warp import warp


def test_cupy_basics():
    try:
        import cupy
        with cupy.cuda.Device(0):
            array = cupy.array([1, 2, 3])
            assert cupy.median(array) == 2
            aprint("\nWorked!")

    except (ModuleNotFoundError, NotImplementedError):
        aprint("Cupy module not found! ignored!")


def test_list_devices():
    try:
        # Test list devices:

        # this will fail if cupy is not available:
        import cupy

        available = CupyBackend.available_devices()
        aprint(f"Available devices: {available}")
        assert len(available) > 0

        for device_id in available:
            with CupyBackend(device_id) as backend:
                aprint(backend)
                xp = Backend.get_xp_module()

                array1 = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)
                array2 = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)

                array1 = Backend.to_backend(array1, numpy.float32)
                array2 = Backend.to_backend(array2, numpy.float32)

    except ModuleNotFoundError:
        aprint("Cupy module not found! Test passes nevertheless!")


def test_allocation_pool():
    try:

        with CupyBackend(enable_memory_pool=True) as root_backend:
            with CupyBackend(enable_memory_pool=True) as backend:
                gc.collect()
                backend.clear_memory_pool()
                gc.collect()

                xp = backend.get_xp_module()

                import cupy

                in_pool_before = backend.mempool.total_bytes() - backend.mempool.used_bytes()
                aprint(f"in_pool_before={in_pool_before}")

                for i in range(10):
                    array = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)

                sleep(2)
                backend.synchronise()

                in_pool_after = backend.mempool.total_bytes() - backend.mempool.used_bytes()
                aprint(f"in_pool_after={in_pool_after}")

                assert in_pool_after > in_pool_before

                backend.clear_memory_pool()

                in_pool_after_clear = backend.mempool.total_bytes() - backend.mempool.used_bytes()
                aprint(f"in_pool_after_clear={in_pool_after_clear}")

                assert in_pool_after_clear <= in_pool_before

                for i in range(10):
                    array = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)

                in_pool_after_reallocation = backend.mempool.total_bytes() - backend.mempool.used_bytes()
                aprint(f"in_pool_after_reallocation={in_pool_after_reallocation}")

                assert in_pool_after_reallocation > in_pool_after_clear

            in_pool_after_context = root_backend.mempool.total_bytes() - root_backend.mempool.used_bytes()
            aprint(f"in_pool_after_context={in_pool_after_clear}")

            assert in_pool_after_context == in_pool_after_clear


    except ModuleNotFoundError:
        aprint("Cupy module not found! Test passes nevertheless!")


def test_unified_memory():
    try:

        with CupyBackend(enable_memory_pool=True, enable_unified_memory=True):
            xp = Backend.get_xp_module()

            # a lot of arrays...
            arrays = tuple(xp.random.uniform(0, 1, size=(512,) * 3) for i in range(30))

            sum = reduce((lambda x, y: x + y), arrays)

            aprint(sum)

    except ModuleNotFoundError:
        aprint("Cupy module not found! Test passes nevertheless!")


def test_paralell():
    try:
        num_devices = len(CupyBackend.available_devices())
        if num_devices <= 1:
            return

        def f(id):
            with CupyBackend(id):
                aprint(f"Begin: Job on device #{id}")
                xp = Backend.get_xp_module()
                array = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)
                array += 1
                aprint(f"End: Job on device #{id}")

        n_jobs = 2
        Parallel(n_jobs=n_jobs, backend='threading')(delayed(f)(id) for id in range(n_jobs))

    except ModuleNotFoundError:
        aprint("Cupy module not found! Test passes nevertheless!")


def test_paralell_with_exclusive():
    try:
        num_devices = len(CupyBackend.available_devices())
        if num_devices <= 1:
            return

        aprint(f"num_devices = {num_devices}")

        job_duration = 2

        def f(id, device):
            aprint(f"Job #{id} waiting to gain access to device {device} to start ...")
            with CupyBackend(device, exclusive=True):
                with asection(f"Begin: Job #{id} on device #{device}"):
                    xp = Backend.get_xp_module()
                    array = xp.random.uniform(0, 1, size=(128,) * 3).astype(numpy.float32)
                    array += 1
                    sleep(job_duration)
                    aprint(f"End: Job #{id} on device #{device}")

        n_jobs = 2 * num_devices
        aprint(f"n_jobs = {n_jobs}")

        start = time()
        with asection(f"Start jobs"):
            Parallel(n_jobs=n_jobs, backend='threading')(delayed(f)(id, id % num_devices) for id in range(n_jobs))
        elapsed_time = time() - start
        aprint(f"elapsed_time={elapsed_time}")

        # the fact that we have only 'num_devices' available, enforce exclusive access to devices, and spawning more jobs than devices, makes the
        # total elapsed time predictable and testable:
        assert num_devices * job_duration < elapsed_time < (num_devices + 0.5) * job_duration

    except ModuleNotFoundError:
        aprint("Cupy module not found! Test passes nevertheless!")


def test_stress():
    try:
        size = 320

        num_devices = len(CupyBackend.available_devices())
        if num_devices <= 0:
            return

        aprint(f"num_devices = {num_devices}")

        def f(id, device):
            aprint(f"Job #{id} waiting to gain access to device {device} to start ...")
            with CupyBackend(device, exclusive=True, enable_unified_memory=True) as backend:
                with asection(f"Begin: Job #{id} on device #{device}"):
                    xp = Backend.get_xp_module()
                    array = xp.random.uniform(0, 1, size=(size,) * 3)
                    array += xp.random.uniform(0, 1, size=(size,) * 3)

                    # Let's have some texture memory allocated too:
                    vector_field = numpy.random.uniform(low=-5, high=+5, size=(8,) * 3 + (3,))
                    for i in range(10):
                        array = warp(array, vector_field, vector_field_upsampling=4)
                        array += xp.random.uniform(0, 1, size=(size,) * 3)

                    backend.clear_memory_pool()
                    array *= 0.1

                    aprint(f"End: Job #{id} on device #{device}")

        n_jobs = 10 * num_devices
        aprint(f"n_jobs = {n_jobs}")

        with asection(f"Start jobs"):
            Parallel(n_jobs=n_jobs, backend='threading')(delayed(f)(id, id % num_devices) for id in range(n_jobs))


    except ModuleNotFoundError:
        aprint("Cupy module not found! Test passes nevertheless!")
