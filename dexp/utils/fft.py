from dexp.utils.backends import Backend, CupyBackend


def clear_fft_plan_cache() -> None:
    """Clears fft plan cache
    Reference: https://github.com/cupy/cupy/issues/5134#issuecomment-833619964
    """
    backend = Backend.current()
    if not isinstance(backend, CupyBackend):
        return

    from cupy.fft.config import get_plan_cache

    cache = get_plan_cache()
    prev_size = cache.get_size()
    prev_mem = cache.get_memsize()
    cache.set_size(0)

    backend.clear_memory_pool()

    cache.set_size(prev_size)
    cache.set_memsize(prev_mem)
