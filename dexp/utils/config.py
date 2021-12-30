import multiprocessing

from arbol import aprint
from numcodecs import blosc


def config_blosc():
    _cpu_count = multiprocessing.cpu_count() // 2
    _nb_threads = max(1, _cpu_count)
    blosc.use_threads = True
    blosc.set_nthreads(_nb_threads)
    aprint(f"Configured the number of threads used by BLOSC: {blosc.get_nthreads()}")
