from contextlib import contextmanager
from time import time

from arbol import aprint


@contextmanager
def timeit(description: str = "") -> None:
    start = time()
    yield
    elapsed_time = time() - start

    aprint(f"Elapsed time for '{description}': {elapsed_time} seconds")
    return elapsed_time
