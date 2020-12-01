import os
from typing import Any

import numpy

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

    def __init__(self,
                 device_id=0,
                 enable_memory_pool: bool = False,
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
        enable_memory_pool : Enables cupy memory pool. By default disabled, when enabled cupy tends to return out-of-memory exceptions when handling large arrays.
        enable_cub : enables CUB accelerator
        enable_cutensor : enables cuTensor accelerator
        enable_fft_planning : enables FFT planning
        enable_dask_cuda_cluster : enables dask cuda cluster
        enable_dask_cuda_nvlink : enables nvlink
        """

        super().__init__()
        self.device_id = device_id

        import cupy
        self.cupy_device: cupy.cuda.Device = cupy.cuda.Device(self.device_id)

        self.stream = cupy.cuda.stream.Stream(non_blocking=False)

        from cupy.cuda import cub, cutensor
        cub.available = enable_cub
        cutensor.available = enable_cutensor

        if not enable_memory_pool:
            from cupy.cuda import set_allocator, set_pinned_memory_allocator
            # Disable memory pool for device memory (GPU)
            set_allocator(None)
            # Disable memory pool for pinned memory (CPU).
            set_pinned_memory_allocator(None)

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
        return (f"CUDA device id:{self.device_id} "
                f"with {free_mem // (1024 * 1024)} MB ({percent}%) free memory out of {free_mem // (1024 * 1024)} MB, "
                f"compute:{self.cupy_device.compute_capability}, pci-bus-id:'{self.cupy_device.pci_bus_id}'")

    def __enter__(self):
        self.cupy_device.__enter__()
        self.stream.__enter__()
        return super().__enter__()

    def __exit__(self, type, value, traceback):
        super().__exit__(type, value, traceback)
        self.stream.__exit__()
        self.cupy_device.__exit__()
        import cupy
        mempool = cupy.get_default_memory_pool()
        pinned_mempool = cupy.get_default_pinned_memory_pool()
        if mempool is not None:
            mempool.free_all_blocks()
        if pinned_mempool is not None:
            pinned_mempool.free_all_blocks()

    def synchronise(self):
        self.stream.synchronize()

    def close(self):
        # Nothing to do
        pass

    def to_numpy(self, array, dtype=None, force_copy: bool = False) -> numpy.ndarray:
        import cupy
        if cupy.get_array_module(array) == cupy:
            array = cupy.asnumpy(array)

        if dtype:
            return array.astype(dtype, copy=force_copy)
        elif force_copy:
            return array.copy()
        else:
            return array

    def to_backend(self, array, dtype=None, force_copy: bool = False) -> Any:

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
            with cupy.cuda.Device(self.device_id):
                return cupy.asarray(array, dtype=dtype)

    def get_xp_module(self, array=None) -> Any:
        if array is not None:
            import cupy
            return cupy.get_array_module(array)
        else:
            import cupy
            return cupy

    def get_sp_module(self, array=None) -> Any:
        if array is not None:
            import cupyx
            return cupyx.scipy.get_array_module(array)
        else:
            import cupyx
            return cupyx.scipy
