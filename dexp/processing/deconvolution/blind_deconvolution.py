from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from toolz import curry

from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.utils import xpArray
from dexp.utils.backends import Backend


@curry
def _objfun(coefs: ArrayLike, observed: xpArray, deconved: xpArray, psf_fun: Callable) -> float:
    # TODO: deconved could be pre computed in the freq domain

    backend = Backend.current()

    psf = psf_fun(coefs)
    psf = backend.to_backend(psf)

    blurred = fft_convolve(deconved, psf, mode="reflect")
    blurred -= blurred.min()
    blurred /= blurred.max()

    return ((observed - blurred) ** 2).mean().item()


def blind_deconvolution(
    image: xpArray,
    observed_psf: xpArray,
    deconv_fun: Callable[[xpArray, xpArray], xpArray],
    microscope_params: Dict,
    n_iterations: int,
    n_zernikes: int = 15,
    psf_output_path: Optional[str] = None,
    display: bool = False,
) -> xpArray:
    # Local imports to
    from pyotf.phaseretrieval import PhaseRetrievalResult, retrieve_phase
    from pyotf.utils import prep_data_for_PR

    if display:
        import napari

        viewer = napari.Viewer()

    backend = Backend.current()

    image = backend.to_backend(image)

    psf = prep_data_for_PR(observed_psf)
    phase_retriv: PhaseRetrievalResult = retrieve_phase(psf, params=microscope_params)
    phase_retriv.fit_to_zernikes(n_zernikes)

    def get_psf(coefs: Optional[ArrayLike] = None) -> ArrayLike:
        if coefs is not None:
            phase_retriv.zd_result.pcoefs = coefs[: len(coefs) // 2]
            phase_retriv.zd_result.mcoefs = coefs[len(coefs) // 2 :]
        psf = phase_retriv.generate_zd_psf()
        return (psf / psf.sum()).astype(image.dtype)

    if display:
        viewer.add_image(backend.to_numpy(image), name="Input Image")
        viewer.add_image(observed_psf, name="Observed PSF")
        viewer.add_image(psf, name="Prepared PSF")

    psf = get_psf()
    if display:
        viewer.add_image(psf, name="Estimated PSF")

    coefs = np.concatenate((phase_retriv.zd_result.pcoefs, phase_retriv.zd_result.mcoefs))

    for i in range(n_iterations):

        deconv = deconv_fun(image, psf)

        objfun = _objfun(observed=image, deconved=deconv, psf_fun=get_psf)
        opt_res = minimize(objfun, x0=coefs, method="L-BFGS-B")

        old_coefs = coefs
        coefs = opt_res.x
        psf = get_psf(coefs)

        # FIXME
        print(np.square(old_coefs - coefs).mean())

        if display:
            viewer.add_image(backend.to_numpy(deconv), name=f"Deconv i={i}")
            viewer.add_image(psf, name=f"Est. PSF i={i}")

    if display:
        napari.run()

    if psf_output_path is not None:
        np.save(psf_output_path, backend.to_numpy(psf))

    return deconv
