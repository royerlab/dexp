import math
from typing import Tuple, Union

from dexp.processing.filters.butterworth_filter import butterworth_kernel
from dexp.processing.filters.kernels.wiener import wiener_kernel
from dexp.utils.backends import Backend


def wiener_butterworth_kernel(
    kernel,
    alpha: float = 1e-3,
    beta: float = 1e-1,
    cutoffs: Union[float, Tuple[float, ...], None] = None,
    cutoffs_in_freq_units=False,
    auto_cutoff_threshold=0.1,
    order: int = 5,
    dtype=None,
):
    """
    Computes the Wiener-Butterworth back projector according to Guo et al, bioRxiv 2019.

    Parameters
    ----------
    kernel : psf
    alpha : alpha
    beta : beta
    cutoffs : Butterworth cutoffs.
    cutoffs_in_freq_units : If True, the cutoffs are specified in frequency units.
        If False, the units are in normalised within [0,1]
    order : Butterworth order
    dtype : dtype for kernel

    Returns
    -------
    Wiener-Butterworth for given psf.

    """
    backend = Backend.current()
    xp = backend.get_xp_module()

    if dtype is None:
        dtype = kernel.dtype

    wk_f = wiener_kernel(kernel, alpha=alpha, frequency_domain=True, dtype=dtype)

    # TODO: figure out cutoff from PSF ?

    if cutoffs is None:
        cutoffs_in_freq_units = False
        psf_f = xp.log1p(xp.absolute(xp.fft.fftshift(xp.fft.fftn(kernel))))

        psf_sumproj = []
        for i in range(psf_f.ndim):
            s = psf_f.shape[i]
            slicing = (s // 2,) * i + (slice(None),) + (s // 2,) * (psf_f.ndim - 1 - i)
            psf_sumproj.append(psf_f[slicing])

        psf_sumproj = tuple(p / p.max() for p in psf_sumproj)
        psf_sumproj = tuple(p[s // 2 :] for s, p in zip(psf_f.shape, psf_sumproj))
        pass_band = tuple(p > auto_cutoff_threshold for p in psf_sumproj)
        cutoffs = tuple(float(xp.count_nonzero(b) / b.size) for b in pass_band)
        # cutoffs = (max(cutoffs),)*psf_f.ndim

    epsilon = math.sqrt((beta**-2) - 1)

    bwk_f = butterworth_kernel(
        shape=kernel.shape,
        cutoffs=cutoffs,
        cutoffs_in_freq_units=cutoffs_in_freq_units,
        epsilon=epsilon,
        order=order,
        frequency_domain=True,
        dtype=dtype,
    )

    # Weiner-Butterworth back projector
    wbwk_f = wk_f * bwk_f
    wbwk = xp.real(xp.fft.ifftn(wbwk_f))
    wbwk = xp.clip(wbwk, a_min=0, a_max=None)
    wbwk /= wbwk.sum()

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(xp.absolute(xp.fft.fftshift(array)))
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(wk_f), name='wk_f', colormap='viridis')
    #     viewer.add_image(_c(bwk_f), name='bwk_f', colormap='viridis')
    #     viewer.add_image(_c(wbwk_f), name='wbwk_f', colormap='viridis')
    #     viewer.grid_view(2, 2, 1)

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
