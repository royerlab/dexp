import gc

import numpy
from arbol.arbol import asection, aprint, section

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.equalise.equalise_intensity import equalise_intensity
from dexp.processing.filters.butterworth_filter import butterworth_filter
from dexp.processing.fusion.dct_fusion import fuse_dct_nd
from dexp.processing.fusion.dft_fusion import fuse_dft_nd
from dexp.processing.fusion.tg_fusion import fuse_tg_nd
from dexp.processing.multiview_lightsheet.fusion.simview import fuse_illumination_views
from dexp.processing.registration.model.pairwise_reg_model import PairwiseRegistrationModel
from dexp.processing.registration.reg_trans_nd import register_translation_nd
from dexp.processing.registration.reg_trans_nd_maxproj import register_translation_maxproj_nd
from dexp.processing.registration.reg_warp_multiscale_nd import register_warp_multiscale_nd
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions
from dexp.processing.restoration.dehazing import dehaze


@section("mvSOLS 1C2L fusion")
def msols_fuse_1C2L(C0L0, C0L1,
                    dz: float,
                    dx: float,
                    angle: float,
                    resampling_mode: str = 'yang',
                    equalise: bool = True,
                    zero_level: float = 120,
                    clip_too_high: int = 2048,
                    fusion='tg',
                    fusion_bias_exponent: int = 2,
                    fusion_bias_strength_x: float = 0.1,
                    z_pad: int = 0,
                    z_apodise: int = 0,
                    registration_num_iterations: int = 4,
                    registration_confidence_threshold: float = 0.3,
                    registration_max_residual_shift: int = 32,
                    registration_mode: str = 'projection',
                    registration_edge_filter: bool = False,
                    registration_force_model: bool = False,
                    registration_model: PairwiseRegistrationModel = None,
                    registration_min_confidence: float = 0.5,
                    registration_max_change: int = 16,
                    dehaze_before_fusion: bool = True,
                    dehaze_size: int = 65,
                    dark_denoise_threshold: int = 0,
                    dark_denoise_size: int = 9,
                    butterworth_filter_cutoff: float = 1,
                    internal_dtype=numpy.float16):
    """

    Parameters
    ----------
    C0L0 : Image for Camera 0 lightsheet 0
    C0L1 : Image for Camera 0 lightsheet 1

    dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)

    dx     : float, pixel size of the camera

    angle  : float, incident angle of the light sheet, angle between the light sheet and the optical axis

    mode : Resampling mode, can be 'byang' for Bin Yang's resampling ;-)

    equalise : Equalise intensity of views before fusion, or not.

    zero_level : Zero level: that's the minimal detector pixel value floor to substract,
    typically for sCMOS cameras the floor is at around 100 (this is to avoid negative values
    due to electronic noise!). Substracting a bit more than that is a good idea to clear out noise
    in the background --  hence the default of 120.

    clip_too_high : clips very high intensities, to avoid loss of precision when converting an internal format such as float16

    fusion : Fusion mode, can be 'tg', 'dct', 'dft'

    fusion_bias_exponent : Exponent for fusion bias

    fusion_bias_strength_x : Strength of fusion bias for fusing views along x axis (after resampling/deskewing) . Set to zero to deactivate

    z_pad : Padding length along Z (scanning direction) for input stacks, usefull in conjunction with z_apodise

    z_apodise : apodises along Z (direction) to suppress discontinuities (views cropping the sample) that disrupt fusion.

    registration_confidence_threshold : Confidence threshold within [0, 1]: zero means low confidence, 1 max confidence.

    registration_max_residual_shift : After teh first registration round, if a shift vector norm is larger than this provided limitRegistration

    registration_mode : Registration mode, can be: 'projection' or 'full'.
    Projection mode is faster but might have occasionally  issues for certain samples. Full mode is slower and is only recomended as a last resort.

    registration_edge_filter : apply edge filter to help registration

    registration_force_model : Forces the use of the provided model (see below)

    registration_model : registration model to use the two camera views (C0Lx and C1Lx),
    if None, the two camera views are registered, and the registration model is returned.

    registration_min_confidence : Minimal confidence for registration parameters, if below that level the registration parameters for previous time points is used.

    registration_max_change : Maximal change in registration parameters, if above that level the registration parameters for previous time points is used.

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
    sp = Backend.get_sp_module()

    if C0L0.dtype != C0L1.dtype:
        raise ValueError("The two views must have same dtype!")

    if C0L0.shape != C0L1.shape:
        raise ValueError("The two views must have same shapes!")

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

    if z_pad > 0 or z_apodise > 0:
        with asection(f"Pad and apodise C0L0 and C0L1 along scanning direction:"):

            if z_pad > 0:
                C0L0[0] = sp.ndimage.gaussian_filter(C0L0[0], sigma=4)
                C0L0[-1] = sp.ndimage.gaussian_filter(C0L0[-1], sigma=4)
                C0L0 = xp.pad(C0L0, pad_width=((z_pad, z_pad),) + ((0, 0),) * 2, mode='edge')

                C0L1[0] = sp.ndimage.gaussian_filter(C0L1[0], sigma=4)
                C0L1[-1] = sp.ndimage.gaussian_filter(C0L1[-1], sigma=4)
                C0L1 = xp.pad(C0L1, pad_width=((z_pad, z_pad),) + ((0, 0),) * 2, mode='edge')

            if z_apodise > 0:
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

    # from napari import gui_qt, Viewer
    # with gui_qt():
    #     def _c(array):
    #         return Backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(C0L0), name='C0L0', contrast_limits=(0, 1000))
    #     viewer.add_image(_c(C0L1), name='C0L1', contrast_limits=(0, 1000))

    with asection(f"Resample C0L0 and C0L1"):
        C0L0 = resample_C0L0(C0L0, angle=angle, dx=dx, dz=dz, mode=resampling_mode)
        C0L1 = resample_C0L1(C0L1, angle=angle, dx=dx, dz=dz, mode=resampling_mode)
        aprint(f"Shape and dtype of C0L0 after resampling: {C0L0.shape}, {C0L0.dtype}")
        aprint(f"Shape and dtype of C0L1 after resampling: {C0L1.shape}, {C0L1.dtype}")
        Backend.current().clear_allocation_pool()

    if dehaze_size > 0 and dehaze_before_fusion:
        with asection(f"Dehaze C0L0 & C0L1 ..."):
            C0L0 = dehaze(C0L0,
                          size=dehaze_size,
                          minimal_zero_level=zero_level,
                          correct_max_level=True)
            C0L1 = dehaze(C0L1,
                          size=dehaze_size,
                          minimal_zero_level=zero_level,
                          correct_max_level=True)
            Backend.current().clear_allocation_pool()

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return Backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(C0L0), name='C0L0', colormap='bop blue', blending='additive')
    #     viewer.add_image(_c(C0L1), name='C0L1', colormap='bop orange', blending='additive')

    with asection(f"Register C0L0 and C0L1"):

        aprint(f"Provided registration model: {registration_model}, overall confidence: {0 if registration_model is None else registration_model.overall_confidence()}")

        if registration_force_model and registration_model is not None:
            model = registration_model
        else:
            aprint("No registration model enforced, running registration now")
            registration_method = register_translation_maxproj_nd if registration_mode == 'projection' else register_translation_nd
            new_model = register_warp_multiscale_nd(C0L0, C0L1,
                                                    num_iterations=registration_num_iterations,
                                                    confidence_threshold=registration_confidence_threshold,
                                                    max_residual_shift=registration_max_residual_shift,
                                                    edge_filter=registration_edge_filter,
                                                    registration_method=registration_method,
                                                    denoise_input_sigma=1)

            Backend.current().clear_allocation_pool()
            aprint(f"Computed registration model: {new_model}, overall confidence: {new_model.overall_confidence()}")

            if registration_model is None or (new_model.overall_confidence() >= registration_min_confidence or new_model.change_relative_to(registration_model) <= registration_max_change):
                model = new_model
            else:
                model = registration_model

        aprint(f"Applying registration model: {model}, overall confidence: {model.overall_confidence()}")
        C0L0, C0L1 = model.apply(C0L0, C0L1)
        Backend.current().clear_allocation_pool()

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return Backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(C0L0), name='C0L0', colormap='bop blue', blending='additive')
    #     viewer.add_image(_c(C0L1), name='C0L1', colormap='bop orange', blending='additive')

    if equalise:
        with asection(f"Equalise intensity of C0L0 relative to C0L1 ..."):
            C0L0, C0L1, ratio = equalise_intensity(C0L0, C0L1,
                                                   zero_level=0 if dehaze_before_fusion else zero_level,
                                                   copy=False)

            aprint(f"Equalisation ratio: {ratio}")

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return Backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(C0L0), name='C0L0', colormap='bop blue', blending='additive')
    #     viewer.add_image(_c(C0L1), name='C0L1', colormap='bop orange', blending='additive')

    with asection(f"Fuse detection views C0lx and C1Lx..."):
        C1Lx = fuse_illumination_views(C0L0, C0L1,
                                       mode=fusion,
                                       bias_exponent=fusion_bias_exponent,
                                       bias_strength=fusion_bias_strength_x)
        Backend.current().clear_allocation_pool()

    if dehaze_size > 0 and not dehaze_before_fusion:
        with asection(f"Dehaze CxLx ..."):
            C1Lx = dehaze(C1Lx,
                          size=dehaze_size,
                          minimal_zero_level=0,
                          correct_max_level=True)
            Backend.current().clear_allocation_pool()

    if dark_denoise_threshold > 0:
        with asection(f"Denoise dark regions of CxLx..."):
            C1Lx = clean_dark_regions(C1Lx,
                                      size=dark_denoise_size,
                                      threshold=dark_denoise_threshold)
            Backend.current().clear_allocation_pool()

    if 0 < butterworth_filter_cutoff < 1:
        with asection(f"Filter output using a Butterworth filter"):
            cutoffs = (butterworth_filter_cutoff,) * C1Lx.ndim
            C1Lx = butterworth_filter(C1Lx, shape=(31, 31, 31), cutoffs=cutoffs, cutoffs_in_freq_units=False)
            Backend.current().clear_allocation_pool()

    with asection(f"Convert back to original dtype..."):
        if original_dtype is numpy.uint16:
            C1Lx = xp.clip(C1Lx, 0, None, out=C1Lx)
        C1Lx = C1Lx.astype(dtype=original_dtype, copy=False)
        Backend.current().clear_allocation_pool()

    gc.collect()
    Backend.current().clear_allocation_pool()

    return C1Lx, model


def fuse_illumination_views(CxL0, CxL1,
                            mode,
                            bias_exponent,
                            bias_strength,
                            smoothing: int = 12):
    if mode == 'tg':
        fused = fuse_tg_nd(CxL0, CxL1,
                           downscale=2,
                           tenengrad_smoothing=smoothing,
                           bias_axis=2,
                           bias_exponent=bias_exponent,
                           bias_strength=bias_strength)
    elif mode == 'dct':
        fused = fuse_dct_nd(CxL0, CxL1)
    elif mode == 'dft':
        fused = fuse_dft_nd(CxL0, CxL1)

    return fused


def resample_C0L0(image,
                  dx: float,
                  dz: float,
                  angle: float,
                  mode: str = 'yang',
                  num_split: int = 4):
    return resample_view(image,
                         flip=True,
                         dx=dx,
                         dz=dz,
                         angle=angle,
                         mode=mode,
                         num_split=num_split)


def resample_C0L1(image,
                  dx: float,
                  dz: float,
                  angle: float,
                  mode: str = 'yang',
                  num_split: int = 4):
    return resample_view(image,
                         flip=False,
                         dx=dx,
                         dz=dz,
                         angle=angle,
                         mode=mode,
                         num_split=num_split)


def resample_view(image,
                  flip: bool,
                  dx: float,
                  dz: float,
                  angle: float,
                  mode: str = 'yang',
                  num_split: int = 4):
    """  Resampling

    Parameters
    ----------
    image  : input image (skewed 3D stack)
    flip   : True for view 0 and False for view 1
    dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)
    dx     : float, pixel size of the camera
    angle  : float, incident angle of the light sheet, angle between the light sheet and the optical axis
    num_split

    Returns
    -------

    """
    xp = Backend.get_xp_module()

    # move to backend:
    image = Backend.to_backend(image)
    if flip:
        image = xp.flip(image, axis=0)

    # rotate the data
    image = xp.rot90(image, k=1, axes=(1, 2))

    # deskew and rotate
    if mode == 'yang':
        image = resampling_vertical_split(image,
                                          dx=dx,
                                          dz=dz,
                                          angle=angle,
                                          num_split=num_split)
    else:
        raise ValueError("Unknown mode: {mode} !")

    # flip along axis x
    if flip:
        image = xp.flip(image, axis=2)

    return image


def resampling_vertical_split(image,
                              dz: float = 1.0,
                              dx: float = 0.2,
                              angle: float = 45,
                              num_split: int = 4):
    """ Same as resampling_vertical but splits the input image so that computation can fit in GPU memory.

    Parameters
    ----------
    image   : input image (skewed 3D stack)
    dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)
    dx     : float, pixel size of the camera
    angle  : float, incident angle of the light sheet, angle between the light sheet and the optical axis
    num_split  : number of splits to break down the data into pieces (along y, axis=2) to fit into the memory of GPU

    Returns
    -------
    Resampled image

    """
    xp = Backend.get_xp_module()

    data_gpu_splits = xp.array_split(image, num_split, axis=1)
    for k in range(num_split):
        data_resampled = resampling_vertical(data_gpu_splits[k], dz, dx, angle=angle)
        if k == 0:
            output = Backend.to_numpy(data_resampled)
        else:
            output = numpy.concatenate((output, Backend.to_numpy(data_resampled)), axis=1)

    return output


def resampling_vertical(image,
                        dz: float,
                        dx: float,
                        angle: float):
    """ Resampling of the image by interpolation along vertical direction.
    Here we assume the dz is integer multiple of dx * cos(angle * pi / 180),
    one can also pre-interpolate the data within along the z' axis if this is not the case

    Parameters
    ----------
    image  : input image (skewed 3D stack)
    dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)
    dx     : float, pixel size of the camera
    angle  : float, incident angle of the light sheet, angle between the light sheet and the optical axis
    num_split  : number of splits to break down the data into pieces (along y, axis=2) to fit into the memory of GPU

    Returns
    -------
    Resampled image


    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    (nz, ny, nx) = image.shape
    dtype = image.dtype

    zres = dz * numpy.sin(angle * xp.pi / 180)
    xres = dx * numpy.cos(angle * xp.pi / 180)

    resample_factor = dz / xres
    resample_factor_int = int(round(resample_factor))

    nz_new, ny_new, nx_new = len(range(0, nx, resample_factor_int)), ny, nx + nz * resample_factor_int
    data_reassign = xp.zeros((nz_new, ny_new, nx_new), xp.int16)

    for x in range(nx):
        x_start = x
        x_end = nz * resample_factor_int + x
        data_reassign[x // resample_factor_int, :, x_start:x_end:resample_factor_int] = image[:, :, x].T
    del image

    # rescale the data, interpolate along z
    data_rescale = sp.ndimage.zoom(data_reassign, zoom=(resample_factor_int, 1, 1), order=1)
    del data_reassign

    data_interp = xp.zeros((nz_new, ny_new, nx_new), dtype)

    for z in range(nz_new):
        for k in range(resample_factor_int):
            data_interp[z, :, k::resample_factor_int] = \
                data_rescale[z * resample_factor_int - k, :, k::resample_factor_int]
    del data_rescale

    # rescale the data, to have voxel the same along x an y;
    # remove the first z slice which has artifacts due to resampling
    image_final = sp.ndimage.zoom(data_interp[1:], zoom=(1, 1, xres / dx), order=1)

    return image_final
