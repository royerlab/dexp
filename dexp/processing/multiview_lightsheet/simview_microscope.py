import gc

from dexp.processing.backends.backend import Backend
from dexp.processing.equalise.equalise_intensity import equalise_intensity
from dexp.processing.filters.butterworth import butterworth_filter
from dexp.processing.fusion.dct_fusion import fuse_dct_nd
from dexp.processing.fusion.dft_fusion import fuse_dft_nd
from dexp.processing.fusion.tg_fusion import fuse_tg_nd
from dexp.processing.registration.reg_trans_nd import register_translation_nd
from dexp.processing.registration.reg_trans_nd_maxproj import register_translation_maxproj_nd
from dexp.utils.timeit import timeit


def simview_fuse_2I2D(backend: Backend, C0L0, C0L1, C1L0, C1L1, mode='tg', model=None, zero_level=100, filter=False):
    with timeit("simview 2I2D fusion"):
        print(f"equalise_intensity...")
        C0L0, C0L1, ratio = equalise_intensity(backend, C0L0, C0L1, zero_level=zero_level)

        gc.collect()
        print(f"equalisation ratio: {ratio}")
        print(f"fuse_lightsheets...")
        C0lx = fuse_lightsheets(backend, C0L0, C0L1, mode=mode)
        #
        # with gui_qt():
        #     def _c(array):
        #         return backend.to_numpy(array)
        #     viewer = Viewer()
        #     viewer.add_image(_c(C0L0), name='C0L0', contrast_limits=(0, 700))
        #     viewer.add_image(_c(C0L1), name='C0L1', contrast_limits=(0, 700))
        #     viewer.add_image(_c(C0lx), name='C0lx', contrast_limits=(0, 700))
        del C0L0
        del C0L1
        gc.collect()

        print(f"equalise_intensity...")
        C1L0, C1L1, ratio = equalise_intensity(backend, C1L0, C1L1, zero_level=zero_level)
        gc.collect()
        print(f"equalisation ratio: {ratio}")
        print(f"fuse_lightsheets...")
        C1lx = fuse_lightsheets(backend, C1L0, C1L1, mode=mode)
        del C1L0
        del C1L1
        gc.collect()

        print(f"equalise_intensity...")
        C0lx, C1lx, ratio = equalise_intensity(backend, C0lx, C1lx, zero_level=0)
        gc.collect()
        print(f"equalisation ratio: {ratio}")

        # with gui_qt():
        #     def _c(array):
        #         return backend.to_numpy(array)
        #     viewer = Viewer()
        #     viewer.add_image(_c(C0lx), name='C0lx', contrast_limits=(0,600))
        #     viewer.add_image(_c(C1lx), name='C1lx', contrast_limits=(0,600))
        #     #viewer.add_image(_c(CxLx), name='CxLx', contrast_limits=(0,600))

        print(f"register_stacks...")
        C0lx, C1lx, model = register_stacks(backend, C0lx, C1lx, model=model)
        print(f"registration model: {model}")
        gc.collect()
        print(f"fuse_cameras...")
        CxLx = fuse_cameras(backend, C0lx, C1lx, mode=mode)
        del C0lx
        del C1lx
        gc.collect()

    if filter:
        print(f"Filter output using a Butterworth filter")
        CxLx = butterworth_filter(backend, CxLx, filter_shape=(17, 17, 17), cutoffs=(0.9, 0.9, 0.9))

    return CxLx, model


def fuse_lightsheets(backend: Backend, CxL0, CxL1, mode: str = 'tg', smoothing=12, bias_strength=0.1):
    if mode == 'tg':
        fused = fuse_tg_nd(backend, CxL0, CxL1, downscale=2, tenenegrad_smoothing=smoothing, bias_axis=2, bias_exponent=2, bias_strength=bias_strength)
    elif mode == 'dct':
        fused = fuse_dct_nd(backend, CxL0, CxL1)
    elif mode == 'dft':
        fused = fuse_dft_nd(backend, CxL0, CxL1)

    return fused


def fuse_cameras(backend: Backend, C0Lx, C1Lx, mode: str = 'tg', smoothing=12, bias_strength=0.1):
    if mode == 'tg':
        fused = fuse_tg_nd(backend, C0Lx, C1Lx, downscale=2, tenenegrad_smoothing=smoothing, bias_axis=0, bias_exponent=2, bias_strength=bias_strength)
    elif mode == 'dct':
        fused = fuse_dct_nd(backend, C0Lx, C1Lx)
    elif mode == 'dft':
        fused = fuse_dft_nd(backend, C0Lx, C1Lx)

    return fused


def register_stacks(backend: Backend, C0Lx, C1Lx, mode='maxproj', integral=True, model=None):

    sp = backend.get_sp_module()
    C0Lx = backend.to_backend(C0Lx)
    C1Lx = backend.to_backend(C1Lx)

    if model is None:

        depth = C0Lx.shape[0]
        crop = depth//4
        C0Lx_c = C0Lx[crop:-crop]
        C1Lx_c = C1Lx[crop:-crop]

        if mode == 'maxproj':
            model = register_translation_maxproj_nd(backend, C0Lx, C1Lx)
        elif mode == 'full':
            model = register_translation_nd(backend, C0Lx_c, C1Lx_c)
            # model.shift_vector*=2


    model.integral = integral
    C0Lx_reg, C1Lx_reg = model.apply(backend, C0Lx, C1Lx)

    return C0Lx_reg, C1Lx_reg, model
