import gc
import os
import threading
from typing import Any

import numpy
from arbol import aprint

from dexp.processing.backends.backend import Backend


class CupyBackend(Backend):
    _dexp_cuda_cluster = None
    _dexp_dask_client = None

    @staticmethod
    def available_devices():
        # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        import GPUtil
        return GPUtil.getAvailable(order='first', limit=numpy.Inf, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])

    device_locks = tuple(threading.Lock() for _ in available_devices.__func__())

    def __init__(self,
                 device_id=0,
                 exclusive: bool = False,
                 enable_streaming: bool = True,
                 enable_memory_pool: bool = True,
                 enable_unified_memory: bool = False,
                 enable_cub: bool = True,
                 enable_cutensor: bool = True,
                 enable_fft_planning: bool = True,
                 enable_dask_cuda_cluster: bool = False,
                 enable_dask_cuda_nvlink: bool = False,
                 ):
        """
        Instantiates a Numpy-based Image Processing backend

        Parameters
        ----------
        device_id : CUDA device id to use for allocation and compute
        exclusive : If True the access to this device is exclusive, no other backend context can access it (when using the context manager idiom)
        enable_memory_pool : Enables cupy memory pool. By default disabled, when enabled cupy tends to return out-of-memory exceptions when handling large arrays.
        enable_cub : enables CUB accelerator
        enable_cutensor : enables cuTensor accelerator
        enable_fft_planning : enables FFT planning
        enable_dask_cuda_cluster : enables dask cuda cluster
        enable_dask_cuda_nvlink : enables nvlink
        """

        super().__init__()
        self.device_id = device_id
        self.exclusive = exclusive
        self.enable_streaming = enable_streaming
        self.enable_unified_memory = enable_unified_memory

        import cupy
        self.cupy_device: cupy.cuda.Device = cupy.cuda.Device(self.device_id)

        from cupy.cuda import cub, cutensor
        cub.available = enable_cub
        cutensor.available = enable_cutensor

        if not enable_memory_pool:
            from cupy.cuda import set_allocator, set_pinned_memory_allocator
            # Disable memory pool for device memory (GPU)
            set_allocator(None)
            # Disable memory pool for pinned memory (CPU).
            set_pinned_memory_allocator(None)
        else:
            self.mempool = None

        if not enable_fft_planning:
            import cupy
            cupy.fft.config.enable_nd_planning = False

        if enable_dask_cuda_cluster and CupyBackend._dexp_cuda_cluster is None:
            from dask_cuda import LocalCUDACluster
            from distributed import Client

            CupyBackend._dexp_cuda_cluster = LocalCUDACluster(enable_nvlink=enable_dask_cuda_nvlink)
            CupyBackend._dexp_dask_client = Client(CupyBackend._dexp_cuda_cluster)

        ## Important: Leave this, this is to make sure that the ndimage package works properly!
        exec("import cupyx.scipy.ndimage")

    def __str__(self):
        free_mem = self.cupy_device.mem_info[0]
        total_mem = self.cupy_device.mem_info[0]
        percent = (100 * free_mem) // total_mem
        return (f"Cupy backend [device id:{self.device_id} "
                f"with {free_mem // (1024 * 1024)} MB ({percent}%) free memory out of {free_mem // (1024 * 1024)} MB, "
                f"compute:{self.cupy_device.compute_capability}, pci-bus-id:'{self.cupy_device.pci_bus_id}']")

    def __enter__(self):
        if self.exclusive:
            CupyBackend.device_locks[self.device_id].acquire(blocking=True)
        self.cupy_device.__enter__()
        if self.enable_streaming:
            import cupy
            self.stream = cupy.cuda.stream.Stream(non_blocking=True)
            self.stream.__enter__()

        import cupy
        self.mempool = cupy.cuda.MemoryPool(cupy.cuda.memory.malloc_managed if self.enable_unified_memory else None)
        from cupy.cuda import memory
        self._previous_allocator = memory._get_thread_local_allocator()
        memory._set_thread_local_allocator(self.mempool.malloc)

        return super().__enter__()

    def __exit__(self, type, value, traceback):
        super().__exit__(type, value, traceback)

        gc.collect()
        self.clear_allocation_pool()
        if self._previous_allocator is not None:
            from cupy.cuda import memory
            memory._set_thread_local_allocator(self._previous_allocator)
        self.mempool = None

        if self.enable_streaming:
            self.stream.__exit__()
        self.cupy_device.__exit__()
        if self.exclusive:
            CupyBackend.device_locks[self.device_id].release()

    def synchronise(self):
        self.stream.synchronize()

    def clear_allocation_pool(self):
        import cupy

        if self.mempool is not None:
            aprint(f"Number of free blocks before release: {self.mempool.n_free_blocks()}, used:{self.mempool.used_bytes() // 1e9}GB, total:{self.mempool.total_bytes() // 1e9}GB ")
            gc.collect()
            self.mempool.free_all_blocks()
            aprint(f"Number of free blocks after release: {self.mempool.n_free_blocks()}, used:{self.mempool.used_bytes() // 1e9}GB, total:{self.mempool.total_bytes() // 1e9}GB ")
        else:
            aprint("Warning: default cupy memory pool is 'None'")

        pinned_mempool = cupy.get_default_pinned_memory_pool()
        if pinned_mempool is not None:
            # aprint(f"Number of free blocks before release: {pinned_mempool.n_free_blocks()}")
            gc.collect()
            pinned_mempool.free_all_blocks()
            # aprint(f"Number of free blocks after release: {pinned_mempool.n_free_blocks()}")
        # else:
        # aprint("Warning: default cupy pinned memory pool is 'None'")

        super().clear_allocation_pool()

    def _to_numpy(self, array, dtype=None, force_copy: bool = False) -> numpy.ndarray:
        import cupy
        if cupy.get_array_module(array) == cupy:
            array = cupy.asnumpy(array)

        if dtype:
            return numpy.asarray(array).astype(dtype, copy=force_copy)
        elif force_copy:
            return numpy.asarray(array.copy())
        else:
            return numpy.asarray(array)

    def _to_backend(self, array, dtype=None, force_copy: bool = False) -> Any:

        import cupy
        if cupy.get_array_module(array) == cupy:
            if dtype:
                return array.astype(dtype, copy=force_copy)
            elif force_copy:
                return array.copy()
            else:
                return array
        else:
            array = self.to_numpy(array)
            with self.cupy_device:
                return cupy.asarray(array, dtype=dtype)

    def _get_xp_module(self, array=None) -> Any:
        if array is not None:
            import cupy
            return cupy.get_array_module(array)
        else:
            import cupy
            return cupy

    def _get_sp_module(self, array=None) -> Any:
        if array is not None:
            import cupyx
            return cupyx.scipy.get_array_module(array)
        else:
            import cupyx
            return cupyx.scipy
