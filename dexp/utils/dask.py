from typing import Sequence, Union

from dask.distributed import Client
from dask_cuda import LocalCUDACluster


def get_dask_client(scheduler_file_or_devices: Union[str, Sequence[int]]) -> Client:

    if isinstance(scheduler_file_or_devices, str):
        client = Client(scheduler_file=scheduler_file_or_devices)

    else:
        cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=scheduler_file_or_devices)
        client = Client(cluster)

    return client
