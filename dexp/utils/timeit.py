import time

from contextlib import contextmanager
from time import time


@contextmanager
def timeit(description: str = '') -> None:
    start = time()
    yield
    elapsed_time = time() - start

    print(f"Elapsed time for '{description}': {elapsed_time} seconds")
    return elapsed_time
