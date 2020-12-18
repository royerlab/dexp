import gc

import numpy
from arbol import asection, section, aprint

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.equalise.equalise_intensity import equalise_intensity
from dexp.processing.filters.butterworth_filter import butterworth_filter
from dexp.processing.fusion.dct_fusion import fuse_dct_nd
from dexp.processing.fusion.dft_fusion import fuse_dft_nd
from dexp.processing.fusion.tg_fusion import fuse_tg_nd
from dexp.processing.registration.model.pairwise_reg_model import PairwiseRegistrationModel
from dexp.processing.registration.reg_trans_nd import register_translation_nd
from dexp.processing.registration.reg_trans_nd_maxproj import register_translation_maxproj_nd
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions
from dexp.processing.restoration.dehazing import dehaze


@section("SimView 2C2L fusion")
def simview_fuse_2C2L(C0L0, C0L1, C1L0, C1L1,
                      equalise: bool = True,
                      zero_level: float = 120,
                      clip_too_high: int = 1024,
                      fusion='tg',
                      fusion_bias_exponent: int = 2,
                      fusion_bias_strength_i: float = 0.5,
                      fusion_bias_strength_d: float = 0.02,
                      registration_mode: str = 'projection',
                      registration_edge_filter: bool = False,
                      registration_crop_factor_along_z: float = 0.3,
                      registration_force_model: bool = False,
                      registration_model: PairwiseRegistrationModel = None,
                      registration_min_confidence: float = 0.5,
                      registration_max_change: int = 16,
                      dehaze_before_fusion: bool = True,
                      dehaze_size: int = 32,
                      dark_denoise_threshold: int = 0,
                      dark_denoise_size: int = 9,
                      butterworth_filter_cutoff: float = 1,
                      flip_camera1: bool = True,
                      internal_dtype=numpy.float16):
    """

    Parameters
    ----------
    C0L0 : Image for Camera 0 lightsheet 0
    C0L1 : Image for Camera 0 lightsheet 1
    C1L0 : Image for Camera 1 lightsheet 0
    C1L1 : Image for Camera 1 lightsheet 1

    equalise : Equalise intensity of views before fusion, or not.

    zero_level : Zero level: that's the minimal detector pixel value floor to substract,
    typically for sCMOS cameras the floor is at around 100 (this is to avoid negative values
    due to electronic noise!). Substracting a bit more than that is a good idea to clear out noise
    in the background --  hence the default of 120.

    clip_too_high : clips very high intensities, to avoid loss of precision when converting an internal format such as float16

    fusion : Fusion mode, can be 'tg', 'dct', 'dft'

    fusion_bias_exponent : Exponent for fusion bias

    fusion_bias_strength_i, fusion_bias_strength_d : Strength of fusion bias for fusing views of different _i_llumination, and of different _d_etections. Set to zero to deactivate

    registration_mode : Registration mode, can be: 'projection' or 'full'.
    Projection mode is faster but might have occasionally  issues for certain samples. Full mode is slower and is only recomended as a last resort.

    registration_edge_filter : apply edge filter to help registration

    registration_crop_factor_along_z : How much to center-crop along z to estimate registration parameters between the two fused camera views. A value of 0.3 means cropping by 30% on both ends.

    registration_force_model : Forces the use of the provided model (see below)

    registration_model : Suggested registration model to use the two camera views (C0Lx and C1Lx) -- typically from a previous time-point or another wavelength.
    Used only if its confidence score is higher than that of the computed registration parameters.

    registration_min_confidence : Minimal confidence for registration parameters, if below that level the registration parameters for previous time points is used.

    registration_max_displacement : Maximal change in registration parameters, if above that level the registration parameters for previous time points is used.

    dehaze_before_fusion : Whether to dehaze the views before fusion or to dehaze the fully fused and registered final image.

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
    xp = Backend.get_xp_module()

    if C0L0.dtype != C0L1.dtype or C0L0.dtype != C1L0.dtype or C0L0.dtype != C1L1.dtype:
        raise ValueError("The four views must have same dtype!")

    if C0L0.shape != C0L1.shape or C0L0.shape != C1L0.shape or C0L0.shape != C1L1.shape:
        raise ValueError("The four views must have same shapes!")

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = C0L0.dtype

    with asection(f"Moving C0L0 and C0L1 to backend storage and converting to {internal_dtype}..."):
        C0L0 = Backend.to_backend(C0L0, dtype=internal_dtype, force_copy=False)
        C0L1 = Backend.to_backend(C0L1, dtype=internal_dtype, force_copy=False)
        Backend.current().clear_allocation_pool()

    if clip_too_high > 0:
        with asection(f"Clipping intensities above {clip_too_high} for C0L0 & C0L1"):
            C0L0 = xp.clip(C0L0, a_min=0, a_max=clip_too_high, out=C0L0)
            C0L1 = xp.clip(C0L1, a_min=0, a_max=clip_too_high, out=C0L1)
            Backend.current().clear_allocation_pool()

    if equalise:
        with asection(f"Equalise intensity of C0L0 relative to C0L1 ..."):
            C0L0, C0L1, ratio = equalise_intensity(C0L0, C0L1,
                                                   zero_level=zero_level,
                                                   copy=False)
            Backend.current().clear_allocation_pool()
            aprint(f"Equalisation ratio: {ratio}")

    if dehaze_size > 0 and dehaze_before_fusion:
        with asection(f"Dehaze C0L0 and C0L1 ..."):
            C0L0 = dehaze(C0L0.copy(),
                          size=dehaze_size,
                          minimal_zero_level=0,
                          correct_max_level=True)
            C0L1 = dehaze(C0L1.copy(),
                          size=dehaze_size,
                          minimal_zero_level=0,
                          correct_max_level=True)

            # from napari import gui_qt, Viewer
            # with gui_qt():
            #     def _c(array):
            #         return Backend.to_numpy(array)
            #
            #     viewer = Viewer()
            #     viewer.add_image(_c(C0L0), name='C0L0', contrast_limits=(0, 1000))
            #     viewer.add_image(_c(C0L0_dh), name='C0L0_dh', contrast_limits=(0, 1000))
            #     viewer.add_image(_c(C0L1), name='C0L1', contrast_limits=(0, 1000))
            #     viewer.add_image(_c(C0L1_dh), name='C0L1_dh', contrast_limits=(0, 1000))
            #
            # C0L0 = C0L0_dh
            # C0L1 = C0L1_dh
            Backend.current().clear_allocation_pool()

    with asection(f"Fuse illumination views C0L0 and C0L1..."):
        C0lx = fuse_illumination_views(C0L0, C0L1,
                                       mode=fusion,
                                       bias_exponent=fusion_bias_exponent,
                                       bias_strength=fusion_bias_strength_i)

        # from napari import gui_qt, Viewer
        # with gui_qt():
        #     def _c(array):
        #         return Backend.to_numpy(array)
        #
        #     viewer = Viewer()
        #     viewer.add_image(_c(C0L0), name='C0L0', contrast_limits=(0, 1000))
        #     viewer.add_image(_c(C0L1), name='C0L1', contrast_limits=(0, 1000))
        #     viewer.add_image(_c(C0lx), name='C0lx', contrast_limits=(0, 1000))

        del C0L0
        del C0L1
        Backend.current().clear_allocation_pool()

    with asection(f"Moving C1L0 and C1L1 to backend storage and converting to {internal_dtype}..."):
        C1L0 = Backend.to_backend(C1L0, dtype=internal_dtype, force_copy=False)
        if flip_camera1:
            C1L0 = xp.flip(C1L0, -1)
        C1L1 = Backend.to_backend(C1L1, dtype=internal_dtype, force_copy=False)
        if flip_camera1:
            C1L1 = xp.flip(C1L1, -1)
        Backend.current().clear_allocation_pool()

    if clip_too_high > 0:
        with asection(f"Clipping intensities above {clip_too_high} for C0L0 & C0L1"):
            C1L0 = xp.clip(C1L0, a_min=0, a_max=clip_too_high, out=C1L0)
            C1L1 = xp.clip(C1L1, a_min=0, a_max=clip_too_high, out=C1L1)
            Backend.current().clear_allocation_pool()

    if equalise:
        with asection(f"Equalise intensity of C1L0 relative to C1L1 ..."):
            C1L0, C1L1, ratio = equalise_intensity(C1L0, C1L1,
                                                   zero_level=zero_level,
                                                   copy=False)
            Backend.current().clear_allocation_pool()
            aprint(f"Equalisation ratio: {ratio}")

    if dehaze_size > 0 and dehaze_before_fusion:
        with asection(f"Dehaze C1L0 and C1L1 ..."):
            C1L0 = dehaze(C1L0,
                          size=dehaze_size,
                          minimal_zero_level=0,
                          correct_max_level=True)
            C1L1 = dehaze(C1L1,
                          size=dehaze_size,
                          minimal_zero_level=0,
                          correct_max_level=True)
            Backend.current().clear_allocation_pool()

    with asection(f"Fuse illumination views C1L0 and C1L1..."):
        C1Lx = fuse_illumination_views(C1L0, C1L1,
                                       mode=fusion,
                                       bias_exponent=fusion_bias_exponent,
                                       bias_strength=fusion_bias_strength_i)

        # from napari import gui_qt, Viewer
        # with gui_qt():
        #     def _c(array):
        #         return Backend.to_numpy(array)
        #
        #     viewer = Viewer()
        #     viewer.add_image(_c(C1L0), name='C1L0', contrast_limits=(0, 1000))
        #     viewer.add_image(_c(C1L1), name='C1L1', contrast_limits=(0, 1000))
        #     viewer.add_image(_c(C1Lx), name='C1Lx', contrast_limits=(0, 1000))

        del C1L0
        del C1L1
        Backend.current().clear_allocation_pool()

    if equalise:
        with asection(f"Equalise intensity of C0lx relative to C1Lx ..."):
            C0lx, C1Lx, ratio = equalise_intensity(C0lx, C1Lx, zero_level=0, copy=False)
            Backend.current().clear_allocation_pool()
            aprint(f"Equalisation ratio: {ratio}")

    with asection(f"Register_stacks C0lx and C1Lx ..."):
        C0lx, C1Lx, registration_model = register_detection_views(C0lx, C1Lx,
                                                                  mode=registration_mode,
                                                                  edge_filter=registration_edge_filter,
                                                                  provided_model=registration_model,
                                                                  force_model=registration_force_model,
                                                                  min_confidence=registration_min_confidence,
                                                                  max_change=registration_max_change,
                                                                  crop_factor_along_z=registration_crop_factor_along_z)
        Backend.current().clear_allocation_pool()

    with asection(f"Fuse detection views C0lx and C1Lx..."):
        CxLx = fuse_detection_views(C0lx, C1Lx,
                                    mode=fusion,
                                    bias_exponent=fusion_bias_exponent,
                                    bias_strength=fusion_bias_strength_d)
        del C0lx
        del C1Lx
        Backend.current().clear_allocation_pool()

    if dehaze_size > 0 and not dehaze_before_fusion:
        with asection(f"Dehaze CxLx ..."):
            CxLx = dehaze(CxLx,
                          size=dehaze_size,
                          minimal_zero_level=0,
                          correct_max_level=True)
            Backend.current().clear_allocation_pool()

    if dark_denoise_threshold > 0:
        with asection(f"Denoise dark regions of CxLx..."):
            CxLx = clean_dark_regions(CxLx,
                                      size=dark_denoise_size,
                                      threshold=dark_denoise_threshold)
            Backend.current().clear_allocation_pool()

    # from napari import gui_qt, Viewer
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(CxLx), name='CxLx', contrast_limits=(0, 1000))
    #     viewer.add_image(_c(CxLx), name='CxLx_dehazed', contrast_limits=(0, 1000))
    #     #viewer.add_image(_c(CxLx_denoised), name='CxLx_denoised', contrast_limits=(0, 1000))

    if 0 < butterworth_filter_cutoff < 1:
        with asection(f"Filter output using a Butterworth filter"):
            cutoffs = (butterworth_filter_cutoff,) * CxLx.ndim
            CxLx = butterworth_filter(CxLx, shape=(31, 31, 31), cutoffs=cutoffs, cutoffs_in_freq_units=False)
            Backend.current().clear_allocation_pool()

    with asection(f"Converting back to original dtype..."):
        if original_dtype is numpy.uint16:
            CxLx = xp.clip(CxLx, 0, None, out=CxLx)
        CxLx = CxLx.astype(dtype=original_dtype, copy=False)
        Backend.current().clear_allocation_pool()

    gc.collect()
    Backend.current().clear_allocation_pool()

    return CxLx, registration_model


def fuse_illumination_views(CxL0, CxL1,
                            mode: str = 'tg',
                            smoothing: int = 12,
                            bias_exponent: int = 2,
                            bias_strength: float = 0.1):
    if mode == 'tg':
        fused = fuse_tg_nd(CxL0, CxL1, downscale=2, tenengrad_smoothing=smoothing, bias_axis=2, bias_exponent=bias_exponent, bias_strength=bias_strength)
    elif mode == 'dct':
        fused = fuse_dct_nd(CxL0, CxL1)
    elif mode == 'dft':
        fused = fuse_dft_nd(CxL0, CxL1)

    return fused


def fuse_detection_views(C0Lx, C1Lx,
                         mode: str = 'tg',
                         smoothing: int = 12,
                         bias_exponent: int = 2,
                         bias_strength: float = 0.1):
    if mode == 'tg':
        fused = fuse_tg_nd(C0Lx, C1Lx, downscale=2, tenengrad_smoothing=smoothing, bias_axis=0, bias_exponent=bias_exponent, bias_strength=bias_strength)
    elif mode == 'dct':
        fused = fuse_dct_nd(C0Lx, C1Lx)
    elif mode == 'dft':
        fused = fuse_dft_nd(C0Lx, C1Lx)
    return fused


def register_detection_views(C0Lx, C1Lx,
                             mode='projection',
                             edge_filter=False,
                             integral=True,
                             provided_model=None,
                             force_model=False,
                             min_confidence=0.5,
                             max_change=16,
                             crop_factor_along_z=0.3):
    C0Lx = Backend.to_backend(C0Lx)
    C1Lx = Backend.to_backend(C1Lx)

    aprint(f"Provided registration model: {provided_model}, overall confidence: {0 if provided_model is None else provided_model.overall_confidence()}")

    if force_model and provided_model is not None:
        model = provided_model
    else:

        depth = C0Lx.shape[0]
        crop = int(depth * crop_factor_along_z)
        C0Lx_c = C0Lx[crop:-crop]
        C1Lx_c = C1Lx[crop:-crop]

        if mode == 'projection':
            new_model = register_translation_maxproj_nd(C0Lx_c, C1Lx_c, edge_filter=edge_filter)
        elif mode == 'full':
            new_model = register_translation_nd(C0Lx_c, C1Lx_c, edge_filter=edge_filter)

        new_model.integral = integral

        aprint(f"Computed registration model: {new_model}, overall confidence: {new_model.overall_confidence()}")

        if provided_model is None or (new_model.overall_confidence() >= min_confidence or new_model.change_relative_to(provided_model) <= max_change):
            model = new_model
        else:
            model = provided_model

    aprint(f"Applying registration model: {model}, overall confidence: {model.overall_confidence()}")
    C0Lx_reg, C1Lx_reg = model.apply(C0Lx, C1Lx)
    return C0Lx_reg, C1Lx_reg, model
