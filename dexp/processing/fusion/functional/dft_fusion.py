from dexp.processing.backends.backend import Backend


def fuse_dft_nd(backend: Backend, image_a, image_b, cutoff: float = 0, clip: bool = True):

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

    tranform = lambda x: sp.fft.fftshift(sp.fft.rfftn(x, norm='ortho'))
    itranform = lambda x: sp.fft.irfftn(sp.fft.ifftshift(x), norm='ortho')

    image_a_dft = tranform(image_a)
    image_b_dft = tranform(image_b)

    image_a_dft_abs = xp.absolute(image_a_dft)
    image_b_dft_abs = xp.absolute(image_b_dft)
    image_a_is_max = image_a_dft_abs>image_b_dft_abs

    image_fused_dft = image_a_is_max*image_a_dft + ~image_a_is_max*image_b_dft

    if 0 < cutoff <= 1:
        c = cutoff
        cutoffs = tuple(int(s * c) for s in image_a.shape)
        cutoffs_slice = tuple(slice(s//2-c//2,s//2+c//2) for s, c in zip(image_a.shape, cutoffs))
        image_fused_dft[cutoffs_slice] = ~image_a_is_max[cutoffs_slice] * image_a_dft[cutoffs_slice] \
                                        + image_a_is_max[cutoffs_slice] * image_b_dft[cutoffs_slice]

    image_fused = itranform(image_fused_dft)

    if clip:
        image_fused = xp.clip(image_fused, min_value, max_value)

    return image_fused


