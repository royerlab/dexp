import warnings
from typing import Dict, List, Optional, Tuple

import numpy
import pandas as pd
from arbol import aprint, asection

from dexp.processing.equalise.equalise_intensity import equalise_intensity
from dexp.processing.filters.butterworth_filter import butterworth_filter
from dexp.processing.fusion.dct_fusion import fuse_dct_nd
from dexp.processing.fusion.dft_fusion import fuse_dft_nd
from dexp.processing.fusion.tg_fusion import fuse_tg_nd
from dexp.processing.morphology import area_opening
from dexp.processing.multiview_lightsheet.fusion.basefusion import BaseFusion
from dexp.processing.registration.model import RegistrationModel
from dexp.processing.registration.translation_nd import register_translation_nd
from dexp.processing.registration.translation_nd_proj import (
    register_translation_proj_nd,
)
from dexp.processing.remove_beads import remove_beads_by_threshold
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions
from dexp.processing.restoration.dehazing import dehaze
from dexp.utils import xpArray
from dexp.utils.backends import Backend


class SimViewFusion(BaseFusion):
    def __init__(
        self,
        registration_model: Optional[RegistrationModel],
        equalise: bool,
        equalisation_ratios: List[Optional[float]],
        zero_level: float,
        clip_too_high: int,
        fusion: str,
        fusion_bias_exponent: int,
        fusion_bias_strength_i: float,
        fusion_bias_strength_d: float,
        dehaze_before_fusion: bool,
        dehaze_size: int,
        dehaze_correct_max_level: bool,
        dark_denoise_threshold: int,
        dark_denoise_size: int,
        white_top_hat_size: int,
        white_top_hat_sampling: int,
        remove_beads: bool,
        butterworth_filter_cutoff: float,
        flip_camera1: bool,
        internal_dtype: numpy.dtype = numpy.float16,
        pad: bool = False,
    ):
        super().__init__(
            registration_model,
            equalise,
            equalisation_ratios,
            zero_level,
            clip_too_high,
            fusion,
            dehaze_before_fusion,
            dehaze_size,
            dehaze_correct_max_level,
            dark_denoise_threshold,
            dark_denoise_size,
            butterworth_filter_cutoff,
            internal_dtype,
        )

        self._white_top_hat_size = white_top_hat_size
        self._white_top_hat_sampling = white_top_hat_sampling

        if self._dehaze_size != 0 and self._white_top_hat_size != 0:
            raise ValueError(
                "`dehaze_size` and `white_top_hat_size` cannot be different from"
                f"zero at the same time. Found {self._dehaze_size} and {self._white_top_hat_size}"
            )

        self._fusion_bias_exponent = fusion_bias_exponent
        self._fusion_bias_strength_i = fusion_bias_strength_i
        self._fusion_bias_strength_d = fusion_bias_strength_d
        self._flip_camera1 = flip_camera1
        self._pad = pad
        self._remove_beads = remove_beads

    def _preprocess_single_view(self, view: xpArray, camera: int, lightsheet: int, flip: bool) -> xpArray:
        xp = Backend.get_xp_module()

        with asection(
            f"Moving C{camera}L{lightsheet} and to backend storage and converting to {self._internal_dtype} ..."
        ):
            view = Backend.to_backend(view, dtype=self._internal_dtype, force_copy=False)

        if self._zero_level != 0:
            view = xp.clip(view, self._zero_level, None)
            view -= self._zero_level

        if self._remove_beads:
            with asection(f"Removing beads of C{camera}L{lightsheet}"):
                view = remove_beads_by_threshold(view)

        if self._clip_too_high > 0:
            with asection(f"Clipping intensities above {self._clip_too_high} for C{camera}L{lightsheet}"):
                xp.clip(view, a_min=0, a_max=self._clip_too_high, out=view)

        if self._dehaze_size > 0 and self._dehaze_before_fusion:
            with asection(f"Dehaze C{camera}L{lightsheet} ..."):
                view = dehaze(view, size=self._dehaze_size, minimal_zero_level=0, correct_max_level=True)

        elif self._white_top_hat_size > 0:
            with asection(f"Filtering with White Top Hat transform C{camera}L{lightsheet} ..."):
                view = self._filtered_white_top_hat(view)

        if flip:
            view = xp.flip(view, -1).copy()

        return view

    def _filtered_white_top_hat(self, view: xpArray) -> xpArray:
        # we don't use our area_white_top_hat because we want remove
        # the noise only from the background estimation
        xp = Backend.get_xp_module()
        sp = Backend.get_sp_module()

        # removing salt and pepper noise (morphological opening)
        filtered = sp.ndimage.morphology.grey_opening(view, size=3).astype(numpy.float32)

        # estimating background
        background = area_opening(filtered, self._white_top_hat_size, self._white_top_hat_sampling)

        # white top hat
        view = xp.clip(view - background, a_min=0, a_max=None).astype(view.dtype)

        return view

    def _preprocess_and_fuse_illumination_views(
        self, view0: xpArray, view1: xpArray, flip: bool, camera: int
    ) -> xpArray:

        view0 = self._preprocess_single_view(view0, camera, 0, flip)
        view1 = self._preprocess_single_view(view1, camera, 1, flip)

        if self._equalise:
            with asection(f"Equalise intensity of C{camera}L0 relative to C{camera}L1 ..."):
                view0, view1, ratio = equalise_intensity(
                    view0,
                    view1,
                    zero_level=0,  # already removed when pre-processing single view
                    correction_ratio=self._equalisation_ratios[camera],
                    copy=False,
                )
                aprint(f"Equalisation ratio: {ratio}")
                self._equalisation_ratios[camera] = ratio

        with asection(f"Fuse illumination views C{camera}L0 and C{camera}L1..."):
            fused_view = self._fuse_illumination_views(view0, view1)

        return fused_view

    def preprocess(
        self,
        C0L0: xpArray,
        C0L1: Optional[xpArray] = None,
        C1L0: Optional[xpArray] = None,
        C1L1: Optional[xpArray] = None,
    ) -> Tuple[xpArray, Optional[xpArray]]:

        if C0L1 is not None:
            self._match_input(C0L0, C0L1)
        if C1L0 is not None:
            self._match_input(C0L0, C1L0)
        if C1L1 is not None:
            self._match_input(C0L0, C1L1)

        if C0L1 is None:
            # processing a single light sheet from a single camera
            C0Lx = self._preprocess_single_view(C0L0, camera=0, lightsheet=0, flip=False)
        else:
            C0Lx = self._preprocess_and_fuse_illumination_views(C0L0, C0L1, flip=False, camera=0)

        if C1L0 is None and C1L1 is None:
            # fusing only images from a single camera
            return C0Lx, None

        if C1L1 is None:
            # processing a single light sheet from a single camera
            C1Lx = self._preprocess_single_view(C1L0, camera=1, lightsheet=0, flip=self._flip_camera1)
        else:
            C1Lx = self._preprocess_and_fuse_illumination_views(C1L0, C1L1, flip=self._flip_camera1, camera=1)

        if self._equalise:
            with asection("Equalise intensity of C0Lx relative to C1Lx ..."):
                C0Lx, C1Lx, ratio = equalise_intensity(
                    C0Lx, C1Lx, zero_level=0, correction_ratio=self._equalisation_ratios[2], copy=False
                )
                aprint(f"Equalisation ratio: {ratio}")
                self._equalisation_ratios[2] = ratio

        return C0Lx, C1Lx

    def postprocess(self, CxLx: xpArray) -> xpArray:
        if self._dehaze_size > 0 and not self._dehaze_before_fusion:
            with asection("Dehaze CxLx ..."):
                CxLx = dehaze(
                    CxLx, size=self._dehaze_size, minimal_zero_level=0, correct_max_level=self._dehaze_correct_max_level
                )

        if self._dark_denoise_threshold > 0:
            with asection("Denoise dark regions of CxLx..."):
                CxLx = clean_dark_regions(CxLx, size=self._dark_denoise_size, threshold=self._dark_denoise_threshold)

        if 0 < self._butterworth_filter_cutoff < 1:
            with asection("Filter output using a Butterworth filter"):
                cutoffs = (self._butterworth_filter_cutoff,) * CxLx.ndim
                CxLx = butterworth_filter(CxLx, shape=(31, 31, 31), cutoffs=cutoffs, cutoffs_in_freq_units=False)

        return CxLx

    def fuse(self, C0Lx: xpArray, C1Lx: xpArray, pad: bool) -> xpArray:
        if self._registration_model is None:
            raise RuntimeError("Registration must be computed beforehand.")

        with asection("Register_stacks C0Lx and C1Lx ..."):
            C0Lx, C1Lx = self._registration_model.apply_pair(C0Lx, C1Lx, pad=pad)

        with asection("Fuse detection views C0lx and C1Lx..."):
            CxLx = self._fuse_detection_views(C0Lx, C1Lx)

        return CxLx

    def __call__(
        self,
        C0L0: xpArray,
        C0L1: Optional[xpArray] = None,
        C1L0: Optional[xpArray] = None,
        C1L1: Optional[xpArray] = None,
    ) -> xpArray:

        original_dtype = C0L0.dtype
        xp = Backend.current().get_xp_module()

        C0Lx, C1Lx = self.preprocess(C0L0, C0L1, C1L0, C1L1)
        if C1Lx is None:
            # fusing only two views from two different light sheets
            CxLx = C0Lx
        else:
            CxLx = self.fuse(C0Lx, C1Lx, pad=self._pad)
            CxLx = self.postprocess(CxLx)

        with asection("Converting back to original dtype..."):
            if original_dtype is numpy.uint16:
                CxLx = xp.clip(CxLx, 0, None, out=CxLx)
            CxLx = CxLx.astype(dtype=original_dtype, copy=False)

        return CxLx

    @staticmethod
    def _fuse_views_generic(
        CxL0: xpArray,
        CxL1: xpArray,
        bias_axis: int,
        mode: str,
        smoothing: int,
        bias_exponent: int,
        bias_strength: float,
        downscale: int = 2,
    ) -> xpArray:
        if mode == "tg":
            fused = fuse_tg_nd(
                CxL0,
                CxL1,
                downscale=downscale,
                tenengrad_smoothing=smoothing,
                bias_axis=bias_axis,
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

    def _fuse_illumination_views(self, CxL0: xpArray, CxL1: xpArray, smoothing: int = 12) -> xpArray:
        return self._fuse_views_generic(
            CxL0, CxL1, 2, self._fusion, smoothing, self._fusion_bias_exponent, self._fusion_bias_strength_i
        )

    def _fuse_detection_views(self, C0Lx: xpArray, C1Lx: xpArray, smoothing: int = 12) -> xpArray:
        return self._fuse_views_generic(
            C0Lx, C1Lx, 0, self._fusion, smoothing, self._fusion_bias_exponent, self._fusion_bias_strength_d
        )

    def compute_registration(
        self, C0Lx: xpArray, C1Lx: xpArray, mode: str, edge_filter: bool, crop_factor_along_z: float
    ) -> None:
        C0Lx = Backend.to_backend(C0Lx)
        C1Lx = Backend.to_backend(C1Lx)

        depth = C0Lx.shape[0]
        crop = int(depth * crop_factor_along_z)
        C0Lx_c = C0Lx[crop:-crop]
        C1Lx_c = C1Lx[crop:-crop]

        if mode == "projection":
            self._registration_model = register_translation_proj_nd(C0Lx_c, C1Lx_c, edge_filter=edge_filter)
        elif mode == "full":
            self._registration_model = register_translation_nd(C0Lx_c, C1Lx_c, edge_filter=edge_filter)
        else:
            raise NotImplementedError

        aprint(
            f"Applying registration model: {self._registration_model},"
            f"overall confidence: {self._registration_model.overall_confidence()}"
        )

    @staticmethod
    def validate_views(views: Dict[str, xpArray]) -> Dict[str, xpArray]:
        if not isinstance(views, dict):
            raise ValueError("Views must be of type Dict[str, xpArray]")

        if len(views) == 4:
            return views
        assert len(views) == 2, f"Found only {views.keys()}, expected 2 or 4 views."
        warnings.warn(f"Only two views found. Fusing according to channels' suffixes {views.keys()}")

        if all("C1" in k for k in views):
            warnings.warn("Swapping names from C1 to C0")
            return {k.replace("C1", "C0"): v for k, v in views.items()}

        if all("L1" in k for k in views):
            warnings.warn("Swapping names from L1 to L0")
            return {k.replace("L1", "L0"): v for k, v in views.items()}

        return views


def summary_from_simview_models(models: List[SimViewFusion]) -> pd.DataFrame:
    df = []
    for m in models:
        current = {}
        for i, eq_ratio in enumerate(m._equalisation_ratios):
            if eq_ratio is not None:
                current[f"eq_ratio_{i}"] = eq_ratio

        if m.registration_model is not None:
            for i, v in enumerate(m.registration_model.shift_vector):
                current[f"shift_{i}"] = v
            current["confidence"] = m.registration_model.overall_confidence()

    return pd.DataFrame(df)
