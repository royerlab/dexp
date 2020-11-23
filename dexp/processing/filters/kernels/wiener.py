from dexp.processing.backends.backend import Backend


def wiener_kernel(backend: Backend,
                  psf,
                  alpha: float = 1e-3,
                  frequency_domain: bool = False,
                  dtype = None):

    xp = backend.get_xp_module()

    if dtype is None:
        dtype = psf.dtype

    psf_fft = xp.fft.fftn(psf)
    kernel_fft = psf_fft/(xp.abs(psf_fft) * xp.abs(psf_fft) + alpha)

    if frequency_domain:
        kernel_fft = kernel_fft.astype(dtype, copy=False)
        return kernel_fft
    else:
        kernel = xp.real(xp.fft.ifftn(kernel_fft))
        kernel = kernel / kernel.sum()
        kernel = kernel.astype(dtype, copy=False)
        return kernel