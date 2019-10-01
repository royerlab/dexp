import time

from contextlib import contextmanager
from time import time


@contextmanager
def timeit(description: str = '') -> None:
    start = time()
    yield
    ellapsed_time = time() - start

    print(f"{description}: {ellapsed_time}")