from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def fuse_tg_nd(backend: Backend, image_a, image_b, cutoff: float = 0, clip: bool = True):

    if not image_a.shape == image_b.shape:
        raise ValueError("Arrays must have the same shape")

    image_a = backend.to_backend(image_a)
    image_b = backend.to_backend(image_b)

    xp = backend.get_xp_module(image_a)
    sp = backend.get_sp_module(image_a)

    min_a, max_a = xp.min(image_a), xp.max(image_a)
    min_b, max_b = xp.min(image_b), xp.max(image_b)
    min_value = min(min_a, min_b)
    max_value = min(max_a, max_b)


    image_fused =

    if clip:
        image_fused = xp.clip(image_fused, min_value, max_value)

    return image_fused





