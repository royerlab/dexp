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
    def available_devices(order='first', maxLoad=1, maxMemory=1):
        """

        Parameters
        ----------
        order :     first --> select the GPU with the lowest ID (DEFAULT)
                    last --> select the GPU with the highest ID
                    random --> select a random available GPU
                    load --> select the GPU with the lowest load
                    memory --> select the GPU with the most memory available
        maxLoad :
        maxMemory

        Returns
        -------

        """
        # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        import GPUtil
        return GPUtil.getAvailable(order=order, limit=numpy.Inf, maxLoad=maxLoad, maxMemory=maxMemory, includeNan=False, excludeID=[], excludeUUID=[])

    device_locks = tuple(threading.Lock() for _ in available_devices.__func__())

    def __init__(self,
                 device_id=0,
                 exclusive: bool = False,
                 enable_streaming: bool = True,
                 enable_memory_pool: bool = True,
                 enable_memory_pool_clearing: bool = True,
                 enable_unified_memory: bool = True,
                 enable_cub: bool = True,
                 enable_cutensor: bool = True,
                 enable_fft_planning: bool = True,
                 enable_dask_cuda_cluster: bool = False,
                 enable_dask_cuda_nvlink: bool = False,
                 ):
        """
        Instantiates a Cupy-based Image Processing backend

        Parameters
        ----------
        device_id : CUDA device id to use for allocation and compute
        exclusive : If True the access to this device is exclusive, no other backend context can access it (when using the context manager idiom)
        enable_memory_pool : Enables cupy memory pool. By default disabled, when enabled cupy tends to return out-of-memory exceptions when handling large arrays.
        enable_memory_pool_clearing : Enables the clearing of the memory pool upon calling 'clear_memory_pool'
        enable_cub : enables CUB accelerator
        enable_cutensor : enables cuTensor accelerator
        enable_fft_planning : enables FFT planning
        enable_dask_cuda_cluster : enables dask cuda cluster
        enable_dask_cuda_nvlink : enables nvlink
        """

        super().__init__()
        self.device_id = device_id
        self.exclusive = exclusive
        self.enable_memory_pool = enable_memory_pool
        self.enable_memory_pool_clearing = enable_memory_pool_clearing
        self.enable_streaming = enable_streaming
        self.enable_unified_memory = enable_unified_memory
        self.mempool = None
        self.stream = None
        self._previous_allocator = None

        import cupy
        self.cupy_device: cupy.cuda.Device = cupy.cuda.Device(self.device_id)

        from cupy.cuda import cub, cutensor
        cub.available = enable_cub
        cutensor.available = enable_cutensor

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

        # lock device:
        if self.exclusive:
            CupyBackend.device_locks[self.device_id].acquire(blocking=True)

        # setup device:
        self.cupy_device.__enter__()

        # setup allocation:
        if self.enable_memory_pool:
            if self.mempool is None:
                import cupy
                from cupy.cuda import set_allocator
                from cupy.cuda.memory import SingleDeviceMemoryPool
                self.mempool = SingleDeviceMemoryPool(cupy.cuda.memory.malloc_managed if self.enable_unified_memory else None)
                from cupy.cuda import memory
            self._previous_allocator = memory._get_thread_local_allocator()
            memory._set_thread_local_allocator(self.mempool.malloc)
        else:
            from cupy.cuda import memory
            memory._set_thread_local_allocator(None)

        # setup stream:
        if self.enable_streaming:
            import cupy
            self.stream = cupy.cuda.stream.Stream(non_blocking=True)
            self.stream.__enter__()

        return super().__enter__()

    def __exit__(self, type, value, traceback):

        super().__exit__(type, value, traceback)

        # unset stream:
        if self.enable_streaming and self.stream is not None:
            self.stream.synchronize()
            self.stream.__exit__()

        # unset allocation:
        self.clear_memory_pool()
        if self._previous_allocator is not None:
            from cupy.cuda import memory
            memory._set_thread_local_allocator(self._previous_allocator)

        # unset device:
        self.cupy_device.__exit__()

        # unlock device:
        if self.exclusive:
            CupyBackend.device_locks[self.device_id].release()

    def synchronise(self):
        if self.stream is not None:
            self.stream.synchronize()

    def clear_memory_pool(self):

        if self.enable_memory_pool_clearing:
            if self.mempool is not None:
                aprint(f"Number of free blocks before release: {self.mempool.n_free_blocks()}, used:{self.mempool.used_bytes() // 1e9}GB, total:{self.mempool.total_bytes() // 1e9}GB ")
                gc.collect()
                self.mempool.free_all_blocks(self.stream)
                aprint(f"Number of free blocks after release: {self.mempool.n_free_blocks()}, used:{self.mempool.used_bytes() // 1e9}GB, total:{self.mempool.total_bytes() // 1e9}GB ")
            else:
                aprint("Warning: default cupy memory pool is 'None'")
        else:
            aprint("Memory pool clearing not enabled!")

        super().clear_memory_pool()

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
