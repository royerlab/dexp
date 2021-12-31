from dexp.utils.backends import Backend


def wiener_kernel(kernel, alpha: float = 1e-3, frequency_domain: bool = False, dtype=None):
    """Computes the Wiener filter for a given kernel and alpha parameter.

    Parameters
    ----------
    kernel : kernel
    alpha : alpha parameter
    frequency_domain : if True then the wiener kernel is returned in the frequency domain
    dtype : dtype for kernel

    Returns
    -------
    Wiener kernel
    """
    backend = Backend.current()
    xp = backend.get_xp_module()

    if dtype is None:
        dtype = kernel.dtype

    psf_fft = xp.fft.fftn(kernel)
    kernel_fft = psf_fft / (xp.abs(psf_fft) * xp.abs(psf_fft) + alpha)

    if frequency_domain:
        return kernel_fft
    else:
        kernel = xp.real(xp.fft.ifftn(kernel_fft))
        kernel = kernel / kernel.sum()
        kernel = kernel.astype(dtype, copy=False)
        return kernel
