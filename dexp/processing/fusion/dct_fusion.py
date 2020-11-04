from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def fuse_dct_nd(backend: Backend,
                image_a,
                image_b,
                cutoff: float = 0,
                clip: bool = True):
    if not image_a.shape == image_b.shape:
        raise ValueError("Arrays must have the same shape")

    if not isinstance(backend, NumpyBackend):
        raise NotImplementedError("DCT not yet implemented in Cupy")

    image_a = backend.to_backend(image_a)
    image_b = backend.to_backend(image_b)

    xp = backend.get_xp_module(image_a)
    sp = backend.get_sp_module(image_a)

    min_a, max_a = xp.min(image_a), xp.max(image_a)
    min_b, max_b = xp.min(image_b), xp.max(image_b)
    min_value = min(min_a, min_b)
    max_value = min(max_a, max_b)

    tranform = lambda x: sp.fft.dctn(x, norm='ortho')
    itranform = lambda x: sp.fft.idctn(x, norm='ortho')

    image_a_dct = tranform(image_a)
    image_b_dct = tranform(image_b)

    image_a_dct_abs = xp.absolute(image_a_dct)
    image_b_dct_abs = xp.absolute(image_b_dct)
    image_a_is_max = image_a_dct_abs > image_b_dct_abs

    image_fused_dct = image_a_is_max * image_a_dct + ~image_a_is_max * image_b_dct

    if 0 < cutoff <= 1:
        c = cutoff
        cutoffs = tuple(int(s * c) for s in image_a.shape)
        cutoffs_slice = tuple(slice(0, c) for c in cutoffs)
        image_fused_dct[cutoffs_slice] = ~image_a_is_max[cutoffs_slice] * image_a_dct[cutoffs_slice] \
                                         + image_a_is_max[cutoffs_slice] * image_b_dct[cutoffs_slice]

    image_fused = itranform(image_fused_dct)

    if clip:
        image_fused = xp.clip(image_fused, min_value, max_value, out=image_fused)

    return image_fused
