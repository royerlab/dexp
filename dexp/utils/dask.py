from typing import Sequence, Union

from dask.distributed import Client
from dask_cuda import LocalCUDACluster


def get_dask_client(address_or_devices: Union[str, Sequence[int]]) -> Client:

    if isinstance(address_or_devices, str):
        client = Client(address=address_or_devices)

    else:
        cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=address_or_devices)
        client = Client(cluster)

    return client
