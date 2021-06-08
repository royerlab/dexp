from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.utils import xpArray


def fuse_dft_nd(image_a: xpArray,
                image_b: xpArray,
                cutoff: float = 0,
                clip: bool = True,
                internal_dtype=None):
    """
    Fuses two images in DFT domain by picking coefficients with maximal magnitude.

    Parameters
    ----------
    image_a : First image to fuse
    image_b : Second image to fuse
    cutoff : frequency cutoff
    clip : Clip fused image to the min and max voxel values of the original images
    -- i.e. the fused image must remain within the bounds of the original images.
    internal_dtype : dtype for internal computation

    Returns
    -------
    Fused image.

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if not image_a.shape == image_b.shape:
        raise ValueError("Arrays must have the same shape")

    if not image_a.dtype == image_b.dtype:
        raise ValueError("Arrays must have the same dtype")

    if internal_dtype is None:
        internal_dtype = image_a.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = xp.float32

    original_dtype = image_a.dtype

    image_a = Backend.to_backend(image_a, dtype=internal_dtype)
    image_b = Backend.to_backend(image_b, dtype=internal_dtype)

    min_a, max_a = xp.min(image_a), xp.max(image_a)
    min_b, max_b = xp.min(image_b), xp.max(image_b)
    min_value = min(min_a, min_b)
    max_value = min(max_a, max_b)

    tranform = lambda x: sp.fft.fftshift(sp.fft.fftn(x, norm='ortho'))
    itranform = lambda x: sp.fft.ifftn(sp.fft.ifftshift(x), norm='ortho')

    image_a_dft = tranform(image_a)
    image_b_dft = tranform(image_b)

    image_a_dft_abs = xp.absolute(image_a_dft)
    image_b_dft_abs = xp.absolute(image_b_dft)
    image_a_is_max = image_a_dft_abs > image_b_dft_abs
    del image_a_dft_abs, image_b_dft_abs

    image_fused_dft = image_a_is_max * image_a_dft
    image_fused_dft += ~image_a_is_max * image_b_dft

    if 0 < cutoff <= 1:
        c = cutoff
        cutoffs = tuple(int(s * c) for s in image_a.shape)
        cutoffs_slice = tuple(slice(s // 2 - c // 2, s // 2 + c // 2) for s, c in zip(image_a.shape, cutoffs))
        image_fused_dft[cutoffs_slice] = ~image_a_is_max[cutoffs_slice] * image_a_dft[cutoffs_slice] \
                                         + image_a_is_max[cutoffs_slice] * image_b_dft[cutoffs_slice]

    image_fused = itranform(image_fused_dft)

    image_fused = xp.real(image_fused)

    if clip:
        image_fused = xp.clip(image_fused, min_value, max_value, out=image_fused)

    # Adjust type:
    image_fused = image_fused.astype(dtype=original_dtype, copy=False)

    return image_fused
