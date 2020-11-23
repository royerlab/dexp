# Derived from Guo et al, bioRxiv 2019.
from typing import Tuple

from dexp.processing.backends.backend import Backend
from dexp.processing.filters.butterworth import butterworth_kernel
from dexp.processing.filters.kernels.wiener import wiener_kernel


def wiener_butterworth_kernel(backend: Backend,
                              psf,
                              alpha: float = 1e-3,
                              beta: float = 1e-3,
                              cutoffs: Tuple[float, ...] = 0.5,
                              order: int = 7,
                              dtype = None):

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    if dtype is None:
        dtype = psf.dtype

    wk_f = wiener_kernel(backend,
                         psf,
                         alpha=alpha,
                         frequency_domain=True,
                         dtype=dtype)

    epsilon = xp.sqrt(1.0 / (beta * beta) - 1)

    ##TODO: figure ot cutoff from PSF ?

    bwk_f = butterworth_kernel(backend,
                               shape=psf.shape,
                               cutoffs=cutoffs,
                               epsilon=epsilon,
                               order=order,
                               frequency_domain=True,
                               dtype=dtype)

    # Weiner-Butterworth back projector
    wbwk_f = wk_f * bwk_f
    wbwk = xp.real(xp.fft.ifftn(wbwk_f))

    wbwk = wbwk.astype(dtype=dtype, copy=False)

    return wbwk

 #
 # ###
 #
 #    pfFFT = np.fft.fft2(pf)
 #
 #    # Wiener-Butterworth back projector.
 #    #
 #    # These values are from Guo et al.
 #    alpha = 0.001
 #    beta = 0.001
 #    n = 8
 #
 #
 #
 #
 #
 #    # Wiener filter
 #    bWiener = pfFFT/(np.abs(pfFFT) * np.abs(pfFFT) + alpha)
 #
 #    # Buttersworth filter
 #    # kv = np.fft.fftfreq(pfFFT.shape[0])
 #    # kx = np.zeros((kv.size, kv.size))
 #    # for i in range(kv.size):
 #    #     kx[i, :] = np.copy(kv)
 #    # ky = np.transpose(kx)
 #    # kk = np.sqrt(kx*kx + ky*ky)
 #
 #    # # This is the cut-off frequency.
 #    # kc = 1.0/(0.5 * 2.355 * sigmaG)
 #    # kkSqr = kk*kk/(kc*kc)
 #    # eps = np.sqrt(1.0/(beta*beta) - 1)
 #    # bBWorth = 1.0/np.sqrt(1.0 + eps * eps * np.power(kkSqr, n))
 #
 #    bw_kernel = butterworth_kernel(backend,
 #                                   frequency_domain=True)
 #
 #    # Weiner-Butterworth back projector
 #    pbFFT = bWiener * bBWorth
 #
 #    # back projector.
 #    pb = np.real(np.fft.ifft2(pbFFT))
 #
 #    return [pf, pb]
