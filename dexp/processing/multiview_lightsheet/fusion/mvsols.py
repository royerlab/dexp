import gc
from typing import Sequence, Tuple

import numpy
from arbol.arbol import aprint, asection, section

from dexp.processing.deskew.yang_deskew import yang_deskew
from dexp.processing.equalise.equalise_intensity import equalise_intensity
from dexp.processing.filters.butterworth_filter import butterworth_filter
from dexp.processing.fusion.dct_fusion import fuse_dct_nd
from dexp.processing.fusion.dft_fusion import fuse_dft_nd
from dexp.processing.fusion.tg_fusion import fuse_tg_nd
from dexp.processing.multiview_lightsheet.fusion.simview import SimViewFusion
from dexp.processing.registration.model.pairwise_registration_model import (
    PairwiseRegistrationModel,
)
from dexp.processing.registration.translation_nd import register_translation_nd
from dexp.processing.registration.translation_nd_proj import (
    register_translation_proj_nd,
)
from dexp.processing.registration.warp_multiscale_nd import register_warp_multiscale_nd
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions
from dexp.processing.restoration.dehazing import dehaze
from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


@section("mvSOLS 1C2L fusion")
def msols_fuse_1C2L(
    C0L0: xpArray,
    C0L1: xpArray,
    dz: float,
    dx: float,
    angle: float,
    resampling_mode: str = "yang",
    equalise: bool = True,
    equalisation_ratios: Sequence[float] = (None,),
    zero_level: float = 120,
    clip_too_high: int = 2048,
    fusion="tg",
    fusion_bias_exponent: int = 2,
    fusion_bias_strength_x: float = 0.1,
    z_pad: int = 0,
    z_apodise: int = 0,
    registration_num_iterations: int = 4,
    registration_confidence_threshold: float = 0.3,
    registration_max_residual_shift: int = 32,
    registration_mode: str = "projection",
    registration_edge_filter: bool = False,
    registration_force_model: bool = False,
    registration_model: PairwiseRegistrationModel = None,
    registration_min_confidence: float = 0.5,
    registration_max_change: int = 16,
    dehaze_before_fusion: bool = True,
    dehaze_size: int = 65,
    dehaze_correct_max_level: bool = True,
    dark_denoise_threshold: int = 0,
    dark_denoise_size: int = 9,
    butterworth_filter_cutoff: float = 1,
    illumination_correction_sigma: float = None,
    huge_dataset_mode: bool = False,
    internal_dtype=numpy.float16,
) -> Tuple:
    """

    Parameters
    ----------
    C0L0 : Image for Camera 0 lightsheet 0
    C0L1 : Image for Camera 0 lightsheet 1

    dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)

    dx     : float, pixel size of the camera

    angle  : float, incident angle of the light sheet, angle between the light sheet and the optical axis

    mode : Resampling mode, can be 'yang' for Bin Yang's resampling ;-)

    equalise : Equalise intensity of views before fusion, or not.

    equalisation_ratios: If provided, these ratios are used instead of calculating
        equalisation values based on the images.
    There is only one equalisation ratio, therefore this is a singleton tuple: (ratio,)

    zero_level : Zero level: that's the minimal detector pixel value floor to substract,
    typically for sCMOS cameras the floor is at around 100 (this is to avoid negative values
    due to electronic noise!). Substracting a bit more than that is a good idea to clear out noise
    in the background --  hence the default of 120.

    clip_too_high : clips very high intensities, to avoid loss of precision when
        converting an internal format such as float16

    fusion : Fusion mode, can be 'tg', 'dct', 'dft'

    fusion_bias_exponent : Exponent for fusion bias

    fusion_bias_strength_x : Strength of fusion bias for fusing views along x axis
        (after resampling/deskewing) . Set to zero to deactivate

    z_pad : Padding length along Z (scanning direction) for input stacks, usefull in conjunction with z_apodise

    z_apodise : apodises along Z (direction) to suppress discontinuities
        (views cropping the sample) that disrupt fusion.

    registration_confidence_threshold : Confidence threshold used for each chunk during warp registration,
        value within [0, 1]: zero means low confidence, 1 max confidence.

    registration_max_residual_shift : After the first registration round of warp registration,
        shift vector with norms larger than this value are deemed low confidence.

    registration_mode : Registration mode, can be: 'projection' or 'full'.
        Projection mode is faster but might have occasionally  issues for certain samples.
        Full mode is slower and is only recomended as a last resort.

    registration_edge_filter : apply edge filter to help registration

    registration_force_model : Forces the use of the provided model (see below)

    registration_model : registration model to use the two camera views (C0Lx and C1Lx),
    if None, the two camera views are registered, and the registration model is returned.

    registration_min_confidence : Minimal confidence for registration parameters,
        if below that level the registration parameters for previous time points is used.

    registration_max_change : Maximal change in registration parameters, if above that level the
        registration parameters for previous time points is used.

    dehaze_before_fusion : Whether to dehaze the views before fusion or to dehaze the fully fused and
        registered final image.

    dehaze_size : After all fusion and registration, the final image is dehazed to remove
    large-scale background light caused by scattered illumination and out-of-focus light.
    This parameter controls the scale of the low-pass filter used.

    dehaze_correct_max_level : Should the dehazing correct the reduced local max intensity
        induced by removing the background?

    dark_denoise_threshold : After all fusion and registration, the final image is processed
    to remove any remaining noise in the dark background region (= hurts compression!).

    dark_denoise_size : Controls the scale over which pixels must be below the threshold to
    be categorised as 'dark background'.

    butterworth_filter_cutoff : At the very end, butterworth_filtering may be applied to
    smooth out spurious high frequencies in the final image. A value of 1 means no filtering,
    a value of e.g. 0.5 means cutting off the top 50% higher frequencies, and keeping all
    frequencies below without any change (that's the point of Butterworth filtering).
    WARNING: Butterworth filtering is currently very slow...

    illumination_correction_sigma: sigma in pixels for correcting the Gaussian profile of a cylindrical lightsheet.

    huge_dataset_mode: optimises memory allocation at the detriment of processing speed to tackle really huge datasets.

    internal_dtype : internal dtype


    Returns
    -------
    Fully registered, fused, dehazed 3D image

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if C0L0.dtype != C0L1.dtype:
        raise ValueError("The two views must have same dtype!")

    if C0L0.shape != C0L1.shape:
        raise ValueError(f"The two views must have same shapes! Found {C0L0.shape} and {C0L1.shape}")

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = C0L0.dtype

    with asection(f"Moving C0L0 and C0L1 to backend storage and converting to {internal_dtype}..."):
        C0L0 = Backend.to_backend(C0L0, dtype=internal_dtype, force_copy=False)
        C0L1 = Backend.to_backend(C0L1, dtype=internal_dtype, force_copy=False)
        Backend.current().clear_memory_pool()

    if clip_too_high > 0:
        with asection(f"Clipping intensities above {clip_too_high} for C0L0 & C0L1"):
            C0L0 = xp.clip(C0L0, a_min=0, a_max=clip_too_high, out=C0L0)
            C0L1 = xp.clip(C0L1, a_min=0, a_max=clip_too_high, out=C0L1)
            Backend.current().clear_memory_pool()

    if z_pad > 0:
        with asection("Pad C0L0 and C0L1 along scanning direction:"):
            C0L0[0] = sp.ndimage.gaussian_filter(C0L0[0], sigma=4)
            C0L0[-1] = sp.ndimage.gaussian_filter(C0L0[-1], sigma=4)
            C0L0 = xp.pad(C0L0, pad_width=((z_pad, z_pad),) + ((0, 0),) * 2, mode="edge")
            Backend.current().clear_memory_pool()

            C0L1[0] = sp.ndimage.gaussian_filter(C0L1[0], sigma=4)
            C0L1[-1] = sp.ndimage.gaussian_filter(C0L1[-1], sigma=4)
            C0L1 = xp.pad(C0L1, pad_width=((z_pad, z_pad),) + ((0, 0),) * 2, mode="edge")
            Backend.current().clear_memory_pool()

    if z_apodise > 0:
        with asection("apodise C0L0 and C0L1 along scanning direction:"):
            depth = C0L0.shape[0]
            apodise_left = xp.linspace(0, 1, num=z_apodise, dtype=internal_dtype)
            apodise_left **= 3
            apodise_center = xp.ones(shape=(depth - 2 * z_apodise,), dtype=internal_dtype)
            apodise_right = xp.linspace(1, 0, num=z_apodise, dtype=internal_dtype) ** 0.5
            apodise_right **= 3
            apodise = xp.concatenate((apodise_left, apodise_center, apodise_right))
            apodise = apodise[:, xp.newaxis, xp.newaxis]
            apodise = apodise.astype(dtype=internal_dtype, copy=False)

            C0L0 *= apodise
            C0L1 *= apodise

            del apodise, apodise_left, apodise_right
            Backend.current().clear_memory_pool()

    # from napari import gui_qt, Viewer
    # with gui_qt():
    #     def _c(array):
    #         return Backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(C0L0), name='C0L0', contrast_limits=(0, 1000))
    #     viewer.add_image(_c(C0L1), name='C0L1', contrast_limits=(0, 1000))

    with asection("Resample C0L0 and C0L1"):
        C0L0 = resample_C0L0(C0L0, angle=angle, dx=dx, dz=dz, mode=resampling_mode)
        aprint(f"Shape and dtype of C0L0 after resampling: {C0L0.shape}, {C0L0.dtype}")
        Backend.current().clear_memory_pool()
        C0L1 = resample_C0L1(C0L1, angle=angle, dx=dx, dz=dz, mode=resampling_mode)
        aprint(f"Shape and dtype of C0L1 after resampling: {C0L1.shape}, {C0L1.dtype}")
        Backend.current().clear_memory_pool()

    if dehaze_size > 0 and dehaze_before_fusion:
        with asection("Dehaze C0L0 & C0L1 ..."):
            C0L0 = dehaze(
                C0L0, size=dehaze_size, minimal_zero_level=zero_level, correct_max_level=dehaze_correct_max_level
            )
            C0L1 = dehaze(
                C0L1, size=dehaze_size, minimal_zero_level=zero_level, correct_max_level=dehaze_correct_max_level
            )
            Backend.current().clear_memory_pool()

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return Backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(C0L0), name='C0L0', colormap='bop blue', blending='additive')
    #     viewer.add_image(_c(C0L1), name='C0L1', colormap='bop orange', blending='additive')

    with asection("Register C0L0 and C0L1"):
        confidence = 0 if registration_model is None else registration_model.overall_confidence()
        aprint(f"Provided registration model: {registration_model}, overall confidence: {confidence}")

        if huge_dataset_mode:
            aprint("Huge dataset mode is on: moving data to CPU RAM.")
            C0L0 = Backend.to_numpy(C0L0)
            C0L1 = Backend.to_numpy(C0L1)

        if registration_force_model and registration_model is not None:
            model = registration_model
        else:
            aprint("No registration model enforced, running registration now")
            registration_method = (
                register_translation_proj_nd if registration_mode == "projection" else register_translation_nd
            )
            new_model = register_warp_multiscale_nd(
                C0L0,
                C0L1,
                num_iterations=registration_num_iterations,
                confidence_threshold=registration_confidence_threshold,
                max_residual_shift=registration_max_residual_shift,
                edge_filter=registration_edge_filter,
                registration_method=registration_method,
                save_memory=huge_dataset_mode,
            )

            Backend.current().clear_memory_pool()
            aprint(f"Computed registration model: {new_model}, overall confidence: {new_model.overall_confidence()}")

            if registration_model is None or (
                new_model.overall_confidence() >= registration_min_confidence
                or new_model.change_relative_to(registration_model) <= registration_max_change
            ):
                model = new_model
            else:
                model = registration_model

        aprint(f"Applying registration model: {model}, overall confidence: {model.overall_confidence()}")
        C0L0, C0L1 = model.apply_pair(C0L0, C0L1)
        Backend.current().clear_memory_pool()

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return Backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(C0L0), name='C0L0', colormap='bop blue', blending='additive')
    #     viewer.add_image(_c(C0L1), name='C0L1', colormap='bop orange', blending='additive')

    if equalise:
        with asection("Equalise intensity of C0L0 relative to C0L1 ..."):
            C0L0, C0L1, ratio = equalise_intensity(
                C0L0,
                C0L1,
                zero_level=0 if dehaze_before_fusion else zero_level,
                correction_ratio=equalisation_ratios[0],
                copy=False,
            )
            equalisation_ratios = (ratio,)
            aprint(f"Equalisation ratio: {ratio}")
            Backend.current().clear_memory_pool()

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return Backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(C0L0), name='C0L0', colormap='bop blue', blending='additive')
    #     viewer.add_image(_c(C0L1), name='C0L1', colormap='bop orange', blending='additive')

    with asection("Fuse detection views C0lx and C1Lx..."):
        C1Lx = SimViewFusion._fuse_views_generic(
            C0L0,
            C0L1,
            bias_axis=2,
            mode=fusion,
            bias_exponent=fusion_bias_exponent,
            bias_strength=fusion_bias_strength_x,
            smoothing=12,
        )
        Backend.current().clear_memory_pool()

    if dehaze_size > 0 and not dehaze_before_fusion:
        with asection("Dehaze CxLx ..."):
            C1Lx = dehaze(C1Lx, size=dehaze_size, minimal_zero_level=0, correct_max_level=dehaze_correct_max_level)
            Backend.current().clear_memory_pool()

    if dark_denoise_threshold > 0:
        with asection("Denoise dark regions of CxLx..."):
            C1Lx = clean_dark_regions(C1Lx, size=dark_denoise_size, threshold=dark_denoise_threshold)
            Backend.current().clear_memory_pool()

    if 0 < butterworth_filter_cutoff < 1:
        with asection("Filter output using a Butterworth filter"):
            cutoffs = (butterworth_filter_cutoff,) * C1Lx.ndim
            C1Lx = butterworth_filter(C1Lx, shape=(31, 31, 31), cutoffs=cutoffs, cutoffs_in_freq_units=False)
            Backend.current().clear_memory_pool()

    if illumination_correction_sigma is not None:
        sigma = illumination_correction_sigma
        length = C1Lx.shape[1]
        correction = xp.linspace(-length // 2, length // 2, num=length)
        correction = xp.exp(-(correction**2) / (2 * sigma * sigma))
        correction = 1.0 / correction
        correction = correction.astype(dtype=internal_dtype)
        C1Lx *= correction[xp.newaxis, :, xp.newaxis]

    with asection("Convert back to original dtype..."):
        if original_dtype is numpy.uint16:
            C1Lx = xp.clip(C1Lx, 0, None, out=C1Lx)
        C1Lx = C1Lx.astype(dtype=original_dtype, copy=False)
        Backend.current().clear_memory_pool()

    gc.collect()
    Backend.current().clear_memory_pool()

    return C1Lx, model, equalisation_ratios


def fuse_illumination_views(CxL0, CxL1, mode, bias_exponent, bias_strength, smoothing: int = 12):
    if mode == "tg":
        fused = fuse_tg_nd(
            CxL0,
            CxL1,
            downscale=2,
            tenengrad_smoothing=smoothing,
            bias_axis=2,
            bias_exponent=bias_exponent,
            bias_strength=bias_strength,
        )
    elif mode == "dct":
        fused = fuse_dct_nd(CxL0, CxL1)
    elif mode == "dft":
        fused = fuse_dft_nd(CxL0, CxL1)
    else:
        raise NotImplementedError

    return fused


def resample_C0L0(image, dx: float, dz: float, angle: float, mode: str = "yang", num_split: int = 8):
    if mode == "yang":
        return yang_deskew(
            image, depth_axis=0, lateral_axis=1, flip_depth_axis=True, dx=dx, dz=dz, angle=angle, num_split=num_split
        )
    else:
        raise ValueError("Invalid deskew mode")


def resample_C0L1(image, dx: float, dz: float, angle: float, mode: str = "yang", num_split: int = 8):
    if mode == "yang":
        return yang_deskew(
            image, depth_axis=0, lateral_axis=1, flip_depth_axis=False, dx=dx, dz=dz, angle=angle, num_split=num_split
        )
    else:
        raise ValueError("Invalid deskew mode")
