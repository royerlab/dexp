import gc

from dexp.processing.backends.backend import Backend
from dexp.processing.equalise.equalise_intensity import equalise_intensity
from dexp.processing.filters.butterworth import butterworth_filter
from dexp.processing.fusion.dct_fusion import fuse_dct_nd
from dexp.processing.fusion.dft_fusion import fuse_dft_nd
from dexp.processing.fusion.tg_fusion import fuse_tg_nd
from dexp.processing.registration.reg_trans_nd import register_translation_nd
from dexp.processing.registration.reg_trans_nd_maxproj import register_translation_maxproj_nd
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions
from dexp.processing.restoration.dehazing import dehaze
from dexp.utils.timeit import timeit


def simview_fuse_2I2D(backend: Backend,
                      C0L0, C0L1, C1L0, C1L1,
                      mode='tg',
                      model=None,
                      zero_level: float = 120,
                      dark_denoise_threshold: int = 80,
                      dehaze_size: int = 65,
                      dark_denoise_size: int = 9,
                      filter=False):
    with timeit("simview 2I2D fusion"):
        print(f"Equalise intensity of C0L0 relative to C0L1 ...")
        C0L0, C0L1, ratio = equalise_intensity(backend, C0L0, C0L1, zero_level=zero_level)
        gc.collect()
        print(f"Equalisation ratio: {ratio}")
        print(f"Fuse illumination views C0L0 and C0L1...")
        C0lx = fuse_illumination_views(backend, C0L0, C0L1, mode=mode)
        del C0L0
        del C0L1
        gc.collect()

        print(f"Equalise intensity of C1L0 relative to C1L1 ...")
        C1L0, C1L1, ratio = equalise_intensity(backend, C1L0, C1L1, zero_level=zero_level)
        gc.collect()
        print(f"Equalisation ratio: {ratio}")
        print(f"Fuse illumination views C1L0 and C1L1...")
        C1lx = fuse_illumination_views(backend, C1L0, C1L1, mode=mode)
        del C1L0
        del C1L1
        gc.collect()

        print(f"Equalise intensity of C0lx relative to C1lx ...")
        C0lx, C1lx, ratio = equalise_intensity(backend, C0lx, C1lx, zero_level=0)
        gc.collect()
        print(f"Equalisation ratio: {ratio}")

        print(f"Register_stacks C0lx and C1lx ...")
        C0lx, C1lx, model = register_views(backend, C0lx, C1lx, model=model)
        print(f"Registration model: {model}")
        gc.collect()
        print(f"Fuse detection views C0lx and C1lx...")
        CxLx = fuse_detection_views(backend, C0lx, C1lx, mode=mode)
        del C0lx
        del C1lx
        gc.collect()

        print(f"Dehaze CxLx ...")
        CxLx = dehaze(backend, CxLx, size=dehaze_size, minimal_zero_level=0)
        gc.collect()

        print(f"Denoise dark regions of CxLx...")
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

    if filter:
        print(f"Filter output using a Butterworth filter")
        CxLx = butterworth_filter(backend, CxLx, filter_shape=(17, 17, 17), cutoffs=(0.9, 0.9, 0.9))

    return CxLx, model


def fuse_illumination_views(backend: Backend, CxL0, CxL1, mode: str = 'tg', smoothing=12, bias_strength=0.1):
    if mode == 'tg':
        fused = fuse_tg_nd(backend, CxL0, CxL1, downscale=2, tenenegrad_smoothing=smoothing, bias_axis=2, bias_exponent=2, bias_strength=bias_strength)
    elif mode == 'dct':
        fused = fuse_dct_nd(backend, CxL0, CxL1)
    elif mode == 'dft':
        fused = fuse_dft_nd(backend, CxL0, CxL1)

    return fused


def fuse_detection_views(backend: Backend, C0Lx, C1Lx, mode: str = 'tg', smoothing=12, bias_strength=0.1):
    if mode == 'tg':
        fused = fuse_tg_nd(backend, C0Lx, C1Lx, downscale=2, tenenegrad_smoothing=smoothing, bias_axis=0, bias_exponent=2, bias_strength=bias_strength)
    elif mode == 'dct':
        fused = fuse_dct_nd(backend, C0Lx, C1Lx)
    elif mode == 'dft':
        fused = fuse_dft_nd(backend, C0Lx, C1Lx)

    return fused


def register_views(backend: Backend, C0Lx, C1Lx, mode='maxproj', integral=True, model=None, crop_factor_along_z=0.3):
    C0Lx = backend.to_backend(C0Lx)
    C1Lx = backend.to_backend(C1Lx)

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
