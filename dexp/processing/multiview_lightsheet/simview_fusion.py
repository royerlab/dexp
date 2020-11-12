import gc

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.equalise.equalise_intensity import equalise_intensity
from dexp.processing.filters.butterworth import butterworth_filter
from dexp.processing.fusion.dct_fusion import fuse_dct_nd
from dexp.processing.fusion.dft_fusion import fuse_dft_nd
from dexp.processing.fusion.tg_fusion import fuse_tg_nd
from dexp.processing.registration.model.pairwise_reg_model import PairwiseRegistrationModel
from dexp.processing.registration.reg_trans_nd import register_translation_nd
from dexp.processing.registration.reg_trans_nd_maxproj import register_translation_maxproj_nd
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions
from dexp.processing.restoration.dehazing import dehaze
from dexp.utils.timeit import timeit


def simview_fuse_2I2D(backend: Backend,
                      C0L0, C0L1, C1L0, C1L1,
                      zero_level: float = 120,
                      fusion='tg',
                      fusion_bias_exponent: int = 2,
                      fusion_bias_strength: float = 0.1,
                      registration_model: PairwiseRegistrationModel = None,
                      dehaze_size: int = 65,
                      dark_denoise_threshold: int = 80,
                      dark_denoise_size: int = 9,
                      butterworth_filter_cutoff: float = 1,
                      internal_dtype=numpy.float16):
    """

    Parameters
    ----------
    backend : Backend to use for computation
    C0L0 : Image for Camera 0 lightsheet 0
    C0L1 : Image for Camera 0 lightsheet 1
    C1L0 : Image for Camera 1 lightsheet 0
    C1L1 : Image for Camera 1 lightsheet 1

    zero_level : Zero level: that's the expected detector pixel vlaue floor to substract,
    typically for sCMOS cameras the floor is at around 100 (this is to avoid negative values
    due to electronic noise!). Substracting a bit more than that is a good idea, hence the
    default of 120.

    fusion : Fusion mode, can be 'tg', 'dct', 'dft'

    registration_model : registration model to use the two camera views (C0Lx and C1Lx),
    if None, the two camera views are registered, and the registration model is returned.

    dehaze_size : After all fusion and registration, the final image is dehazed to remove
    large-scale background light caused by scattered illumination and out-of-focus light.
    This parameter controls the scale of the low-pass filter used.

    dark_denoise_threshold : After all fusion and registration, the final image is processed
    to remove any remaining noise in the dark background region (= hurts compression!).

    dark_denoise_size : Controls the scale over which pixels must be below the threshold to
    be categorised as 'dark background'.

    butterworth_filter_cutoff : At the very end, butterworth_filtering may be applied to
    smooth out spurious high frequencies in the final image. A value of 1 means no filtering,
    a value of e.g. 0.5 means cutting off the top 50% higher frequencies, and keeping all
    frequencies below without any change (that's the point of Butterworth filtering).
    WARNING: Butterworth filtering is currently very slow...

    internal_dtype : internal dtype


    Returns
    -------
    Fully registered, fused, dehazed 3D image

    """

    xp = backend.get_xp_module()

    if C0L0.dtype != C0L1.dtype or C0L0.dtype != C1L0.dtype or C0L0.dtype != C1L1.dtype:
        raise ValueError("The four views must have same dtype!")

    if C0L0.shape != C0L1.shape or C0L0.shape != C1L0.shape or C0L0.shape != C1L1.shape:
        raise ValueError("The four views must have same shapes!")

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    with timeit("SimView 2I2D fusion"):

        original_dtype = C0L0.dtype

        with timeit(f"Moving C0L0 and C0L1 to backend storage and converting to {internal_dtype}..."):
            C0L0 = backend.to_backend(C0L0, dtype=internal_dtype, force_copy=False)
            C0L1 = backend.to_backend(C0L1, dtype=internal_dtype, force_copy=False)
            gc.collect()

        with timeit(f"Equalise intensity of C0L0 relative to C0L1 ..."):
            C0L0, C0L1, ratio = equalise_intensity(backend, C0L0, C0L1, zero_level=zero_level)
            gc.collect()
            print(f"Equalisation ratio: {ratio}")

        with timeit(f"Fuse illumination views C0L0 and C0L1..."):
            C0lx = fuse_illumination_views(backend, C0L0, C0L1, mode=fusion, bias_exponent=fusion_bias_exponent, bias_strength=fusion_bias_strength)
            del C0L0
            del C0L1
            gc.collect()

        with timeit(f"Moving C1L0 and C1L1 to backend storage and converting to {internal_dtype}..."):
            C1L0 = backend.to_backend(C1L0, dtype=internal_dtype, force_copy=False)
            C1L1 = backend.to_backend(C1L1, dtype=internal_dtype, force_copy=False)
            gc.collect()

        with timeit(f"Equalise intensity of C1L0 relative to C1L1 ..."):
            C1L0, C1L1, ratio = equalise_intensity(backend, C1L0, C1L1, zero_level=zero_level)
            gc.collect()
            print(f"Equalisation ratio: {ratio}")

        with timeit(f"Fuse illumination views C1L0 and C1L1..."):
            C1Lx = fuse_illumination_views(backend, C1L0, C1L1, mode=fusion, bias_exponent=fusion_bias_exponent, bias_strength=fusion_bias_strength)
            del C1L0
            del C1L1
            gc.collect()

        with timeit(f"Equalise intensity of C0lx relative to C1Lx ..."):
            C0lx, C1Lx, ratio = equalise_intensity(backend, C0lx, C1Lx, zero_level=0)
            gc.collect()
            print(f"Equalisation ratio: {ratio}")

        with timeit(f"Register_stacks C0lx and C1Lx ..."):
            C0lx, C1Lx, registration_model = register_views(backend, C0lx, C1Lx, model=registration_model)
            print(f"Registration model: {registration_model}")
            gc.collect()

        with timeit(f"Fuse detection views C0lx and C1Lx..."):
            CxLx = fuse_detection_views(backend, C0lx, C1Lx, mode=fusion, bias_exponent=fusion_bias_exponent, bias_strength=fusion_bias_strength)
            del C0lx
            del C1Lx
            gc.collect()

        if dehaze_size > 0:
            with timeit(f"Dehaze CxLx ..."):
                CxLx = dehaze(backend, CxLx, size=dehaze_size, minimal_zero_level=0)
                gc.collect()

        if dark_denoise_threshold > 0:
            with timeit(f"Denoise dark regions of CxLx..."):
                CxLx = clean_dark_regions(backend, CxLx, size=dark_denoise_size,
                                          threshold=dark_denoise_threshold)
                gc.collect()

        # from napari import gui_qt, Viewer
        # with gui_qt():
        #     def _c(array):
        #         return backend.to_numpy(array)
        #     viewer = Viewer()
        #     viewer.add_image(_c(CxLx), name='CxLx', contrast_limits=(0, 1000))
        #     viewer.add_image(_c(CxLx), name='CxLx_dehazed', contrast_limits=(0, 1000))
        #     #viewer.add_image(_c(CxLx_denoised), name='CxLx_denoised', contrast_limits=(0, 1000))

        if 0 < butterworth_filter_cutoff < 1:
            with timeit(f"Filter output using a Butterworth filter"):
                cutoffs = (butterworth_filter_cutoff,) * CxLx.ndim
                CxLx = butterworth_filter(backend, CxLx, filter_shape=(17, 17, 17), cutoffs=cutoffs)
                gc.collect()

        with timeit(f"Converting back to original dtype..."):
            if original_dtype is numpy.uint16:
                CxLx = xp.clip(CxLx, 0, None, out=CxLx)
            CxLx = CxLx.astype(dtype=original_dtype, copy=False)
            gc.collect()

    return CxLx, registration_model


def fuse_illumination_views(backend: Backend,
                            CxL0, CxL1,
                            mode: str = 'tg',
                            smoothing: int = 12,
                            bias_exponent: int = 2,
                            bias_strength: float = 0.1):
    if mode == 'tg':
        fused = fuse_tg_nd(backend, CxL0, CxL1, downscale=2, tenenegrad_smoothing=smoothing, bias_axis=2, bias_exponent=bias_exponent, bias_strength=bias_strength)
    elif mode == 'dct':
        fused = fuse_dct_nd(backend, CxL0, CxL1)
    elif mode == 'dft':
        fused = fuse_dft_nd(backend, CxL0, CxL1)

    return fused


def fuse_detection_views(backend: Backend,
                         C0Lx, C1Lx,
                         mode: str = 'tg',
                         smoothing: int = 12,
                         bias_exponent: int = 2,
                         bias_strength: float = 0.1):
    if mode == 'tg':
        fused = fuse_tg_nd(backend, C0Lx, C1Lx, downscale=2, tenenegrad_smoothing=smoothing, bias_axis=0, bias_exponent=bias_exponent, bias_strength=bias_strength)
    elif mode == 'dct':
        fused = fuse_dct_nd(backend, C0Lx, C1Lx)
    elif mode == 'dft':
        fused = fuse_dft_nd(backend, C0Lx, C1Lx)

    return fused


def register_views(backend: Backend, C0Lx, C1Lx, mode='maxproj', integral=True, model=None, crop_factor_along_z=0.3):
    C0Lx = backend.to_backend(C0Lx)
    C1Lx = backend.to_backend(C1Lx)

    # we need to register if we don't have already a provided model:
    if model is None:
        depth = C0Lx.shape[0]
        crop = int(depth * crop_factor_along_z)
        C0Lx_c = C0Lx[crop:-crop]
        C1Lx_c = C1Lx[crop:-crop]

        if mode == 'maxproj':
            model = register_translation_maxproj_nd(backend, C0Lx_c, C1Lx_c)
        elif mode == 'full':
            model = register_translation_nd(backend, C0Lx_c, C1Lx_c)

        model.integral = integral

    C0Lx_reg, C1Lx_reg = model.apply(backend, C0Lx, C1Lx)

    return C0Lx_reg, C1Lx_reg, model
